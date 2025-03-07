import json
import sys
import string
import numpy as np
import logging
import asyncio
import re
import nemo.collections.asr as nemo_asr

from concurrent.futures import ThreadPoolExecutor
from .vad import remove_non_speech
from .utils import get_language
from stt import (
   logger,
   VAD, VAD_DILATATION, VAD_MIN_SPEECH_DURATION, VAD_MIN_SILENCE_DURATION,
   STREAMING_BUFFER_TRIMMING_SEC, STREAMING_MIN_CHUNK_SIZE,
   STREAMING_PAUSE_FOR_FINAL, STREAMING_TIMEOUT_FOR_SILENCE,
)
from websockets.legacy.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("__streaming__")
logger.setLevel(logging.INFO)

EOF_REGEX = re.compile(r' *\{.*"eof" *: *1.*\} *$')

def bytes_to_array(bytes):
    return np.frombuffer(bytes, dtype=np.int16).astype(np.float32) / 32768

def processor_output_to_text(o):
    if o[0] is None:
        return ""
    return o[2]

def nemo_to_json(o, partial=False):
    result = dict()
    key = "partial" if partial else "text"
    if isinstance(o, list):
        result[key] = ""
        for i in o:
            result[key] += processor_output_to_text(i)
    else:
        result["partial" if partial else "text"] = processor_output_to_text(o)
    json_res = json.dumps(result)
    return json_res


async def wssDecode(ws: WebSocketServerProtocol, model_and_alignementmodel):
    """Async Decode function endpoint"""
    try:
        res = await ws.recv()
        try:
            config = json.loads(res)["config"]
            sample_rate = config["sample_rate"]
            logger.info(f"Received config: {config}")
        except Exception as e:
            logger.error(f"Failed to read stream configuration {e}")
            await ws.close(reason="Failed to load configuration")
        model, _ = model_and_alignementmodel
        language = get_language(config.get("language"))
        streaming_processor = StreamingASRProcessor(model, 
            buffer_trimming=STREAMING_BUFFER_TRIMMING_SEC, pause_for_final=STREAMING_PAUSE_FOR_FINAL,
            vad=VAD, dilatation=VAD_DILATATION, min_silence_duration=VAD_MIN_SILENCE_DURATION, min_speech_duration=VAD_MIN_SPEECH_DURATION
        )
        logger.info("Starting transcription ...")
        executor = ThreadPoolExecutor()
        current_task = None
        received_chunk_size = None
        pile = []
        timeout = None  # it will be computed after the first chunk is received, it is for finding silence in the input stream
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                message = None
            if message and len(pile)<2:
                pile.append(message)
            if (isinstance(message, str) and re.match(EOF_REGEX, message)):
                final = []
                if current_task:    # wait for the last asynchronous prediction to finish
                    o, _ = await current_task
                    final.append(o)
                o, _ = streaming_processor.transcribe()    # make a last prediction in case chunk was too small
                final.append(o)
                logger.debug(f"Last committed text: {o}")
                b = streaming_processor.finish()
                final.append(b)
                logger.debug(f"Last buffered text: {o}")
                await ws.send(nemo_to_json(final))
                await ws.close()
                logger.info("Closing connection")
                break
            if message is None:
                silence_chunk = np.zeros(int(sample_rate * received_chunk_size* 1), dtype=np.float32)
                off = streaming_processor.buffer_time_offset
                dur = len(streaming_processor.audio_buffer)/streaming_processor.sampling_rate
                streaming_processor.insert_audio_chunk(silence_chunk)
                logger.debug(f"Silence chunk inserted ({(len(silence_chunk)/streaming_processor.sampling_rate):.2f}s) at {off:.2f} for {dur:.2f} (now {(len(streaming_processor.audio_buffer)/streaming_processor.sampling_rate):.2f})")
            else:
                audio_chunk = bytes_to_array(message)
                if received_chunk_size is None:
                    received_chunk_size = len(audio_chunk)/sample_rate
                    timeout = received_chunk_size * STREAMING_TIMEOUT_FOR_SILENCE
                streaming_processor.insert_audio_chunk(audio_chunk)
                logger.debug(f"Received chunk of {len(audio_chunk)/sample_rate:.2f}s")
            if streaming_processor.get_buffer_size() >= STREAMING_MIN_CHUNK_SIZE:
                if current_task and not current_task.done():
                    continue
                else:
                    if current_task:    # if the task is done, get the result
                        o, p = await current_task
                        logger.debug(f"Sending final '{o}'")
                        logger.debug(f"Sending partial '{p}'")
                        if o[0] is not None:
                            await ws.send(nemo_to_json(o))
                        else:
                            await ws.send(nemo_to_json(p, partial=True))
                    if len(pile)>0:     # if there are messages in the pile, launch a new transcription task
                        logger.debug(f"Launching new task t={(len(streaming_processor.audio_buffer)/streaming_processor.sampling_rate)+streaming_processor.buffer_time_offset:.2f}s")
                        current_task = asyncio.get_event_loop().run_in_executor(executor, streaming_processor.transcribe)
                        pile.pop(0)
            else:
                logger.debug(f"Chunk too small {streaming_processor.get_buffer_size()}<{STREAMING_MIN_CHUNK_SIZE} (added {len(audio_chunk)/sample_rate}), skipping")
    except ConnectionClosed as e:
        logger.info(f"Connection closed {e}")


class StreamingASRProcessor:


    def __init__(
        self,
        model: nemo_asr.models.EncDecCTCModel,
        buffer_trimming=15,
        pause_for_final=1.5,
        buffer_trimming_words=None,
        vad="auditok",
        logfile=sys.stderr,
        sample_rate=16000,
        min_speech_duration=0.1,
        min_silence_duration=0.1,
        dilatation=0.5,
    ):
        self.model: nemo_asr.models.EncDecCTCModel = model
        self.logfile = logfile
        self.model.eval()
        self.init()

        self.buffer_trimming_sec = buffer_trimming
        self.buffer_trimming_words = buffer_trimming_words
        self.pause_for_final = pause_for_final
        self.vad = vad
        self.vad_dilatation = dilatation
        self.vad_min_speech_duration = min_speech_duration
        self.vad_min_silence_duration = min_silence_duration
        self.sampling_rate = sample_rate
    
    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.commited = []
        self.last_chunked_at = 0

        self.silence_iters = 0
        self.buffered_final = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def get_buffer_size(self):
        return len(self.audio_buffer) / self.sampling_rate
    
    def format_words(self, words, timestamps_convert_function=None):
        word_list = []
        for i in words:
            if timestamps_convert_function is not None:
                start, end = timestamps_convert_function(i['start'], i['end'])
                t = (start, end, i['word'])
            else:
                t = ((i['start'], i['end'], i['word']))
            word_list.append(t)
        return word_list
    
    def get_ends(self, words):
        word_list = []
        for i in words:
            word_list.append(i['end'])
        return word_list

    def transcribe(self):
        if self.vad:
            np_buffer = np.array(self.audio_buffer)
            audio_speech, segments, convertion_function = remove_non_speech(
                np_buffer,
                method=self.vad,
                use_sample=True,
                sample_rate=self.sampling_rate,
                dilatation=self.vad_dilatation,
                min_speech_duration=self.vad_min_speech_duration,
                min_silence_duration=self.vad_min_silence_duration,
            )
            hypothesis = self.model.transcribe([audio_speech], return_hypotheses=True, timestamps=True, verbose=False)[0]
        else:
            hypothesis = self.model.transcribe([self.audio_buffer], return_hypotheses=True, timestamps=True, verbose=False)[0]
        if isinstance(hypothesis, list):
            hypothesis = hypothesis[0]
        formatted_words = self.format_words(hypothesis.timestep['word'], convertion_function if self.vad else None)
        self.transcript_buffer.insert(formatted_words, self.buffer_time_offset)
        o, buffer = self.transcript_buffer.flush()
        self.commited.extend(o)         # contains all text that is commited
        self.buffered_final.extend(o)   # contains text for final
        if (buffer and (self.buffer_time_offset + len(self.audio_buffer) / self.sampling_rate) - buffer[-1][1]< 0.05):
            # remove the last word if it is too close to the end of the buffer
            buffer.pop(-1)
        if len(self.audio_buffer) / self.sampling_rate > self.buffer_trimming_sec:
            self.chunk_completed_segment(
                hypothesis.timestep['word'],
                chunk_silence=self.vad,
                speech_segments=segments if self.vad else False,
            )

        logger.debug(
            f"Len of buffer now: {len(self.audio_buffer)/self.sampling_rate:2.2f}s"
        )
        final = self.make_final(buffer)
        partial = self.buffered_final.copy()
        partial.extend(buffer)
        return final, self.to_flush(partial)


    def make_final(self, buffer):
        final = (None, None, "")
        if len(self.buffered_final)>2:# len(self.buffered_final)>0: # and self.buffered_final[-1][2][-1] in string.punctuation:
            end_word = self.buffered_final[-1][1]
        else:
            end_word = None
        # if there are no words in the buffer (all words are committed), a final should be output
        if len(buffer)==0:
            buffer_end_audio_timestamp = (len(self.audio_buffer)/self.sampling_rate)+self.buffer_time_offset
        # if there are only a few words in the buffer, a final should be output with the text before the buffer
        elif len(buffer)<=3:
            buffer_end_audio_timestamp = buffer[0][1]
        else:
            buffer_end_audio_timestamp = None
        if end_word and buffer_end_audio_timestamp and end_word+self.pause_for_final < buffer_end_audio_timestamp :
            # assemble the final
            f = []
            for i in self.buffered_final:
                if i[1]>end_word:
                    break
                f.append(i)
            if f:
                final = self.to_flush(f)
                self.buffered_final = self.buffered_final[len(f):]
        return final
        
    
    def to_flush(
        self,
        words,
        sep=" ",
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        t = sep.join(s[2] for s in words)
        if len(words) == 0:
            b = None
            e = None
        else:
            b = offset + words[0][0]
            e = offset + words[-1][1]
        return (b, e, t)

    def chunk_completed_segment(self, res, chunk_silence=False, speech_segments=None):
        # if self.commited == [] and not chunk_silence:
        #     return
        self.buffer_trimming_words = None       # deactivated option - allow to set a limit to the buffer size, over this limit it will cut on last commited word instead of segments
        ends = self.get_ends(res)
        if len(ends) > 1 and self.commited:
            t = self.commited[-1][1]
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            buffer_length = len(self.audio_buffer) / self.sampling_rate
            if e <= t and (self.buffer_trimming_words is None or self.buffer_time_offset+buffer_length - e < self.buffer_trimming_words):
                logger.debug(f"Segment chunked at {e:2.2f}s")# : {ends[1]+ self.buffer_time_offset} ")
                self.chunk_at(e)
                return
            elif self.buffer_trimming_words is not None:
                logger.debug(f"Words chunked at {t:2.2f}")
                self.chunk_at(t-0.5)
                return
        elif chunk_silence:
            lenght = len(self.audio_buffer) / self.sampling_rate
            e = self.buffer_time_offset + lenght - 2
            if speech_segments:
                end_silence = lenght - speech_segments[-1][1]
                if end_silence > 2:
                    logger.debug(f"Silence segment chunked at {e:2.2f}")
                    self.chunk_at(e)
                    return
            elif speech_segments is not None:
                logger.debug(f"Silence segment chunked at {e:2.2f}")
                self.chunk_at(e)
                return
        logger.debug(f"Not enough segments to chunk")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.sampling_rate) :]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        
        self.buffered_final.extend(o)
        f = self.to_flush(self.buffered_final)
        
        logger.debug(f"last, noncommited:{f}")
        return f

class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None
        self.last_buffered_time = -1

        self.logfile = logfile

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            logger.debug(f"removing last {i} words:")
                            for j in range(i):
                                logger.debug(f"\t{self.new.pop(0)}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt.lower().translate(str.maketrans("", "", string.punctuation)) == self.buffer[0][2].lower().translate(str.maketrans("", "", string.punctuation)):
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit, self.buffer

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer