# LinTO-STT-Whisper

LinTO-STT-Whisper is an API for Automatic Speech Recognition (ASR) based on [Whisper models](https://openai.com/research/whisper).

LinTO-STT-Whisper can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector. 

It can be used to do offline or real-time transcriptions.

## Pre-requisites

### Requirements

The transcription service requires [docker](https://www.docker.com/products/docker-desktop/) up and running.

For GPU capabilities, it is also needed to install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Hardware

To run the transcription models you'll need:
* At least 8GB of disk space to build the docker image
  and models can occupy several GB of disk space depending on the model size (it can be up to 5GB).
* Up to 7GB of RAM depending on the model used.
* One CPU per worker. Inference time scales on CPU performances.

On GPU, approximate VRAM peak usage are indicated in the following table
for some model sizes, depending on the backend
(note that the lowest precision supported by the GPU card is automatically chosen when loading the model).
<table border="0">
 <tr>
    <td rowspan="3"><b>Model size</b></td>
    <td colspan="4"><b>Backend and precision</b></td>
 </tr>
 <tr>
    <td colspan="3"><b> [ct2/faster_whisper](whisper/Dockerfile.ctranslate2) </b></td>
    <td><b> [torch/whisper_timestamped](whisper/Dockerfile.torch) </b></td>
 </tr>
 <tr>
    <td><b>int8</b></td>
    <td><b>float16</b></td>
    <td><b>float32</b></td>
    <td><b>float32</b></td>
 </tr>
 <tr>
    <td>tiny</td>
    <td colspan="3">1.5G</td>
    <td>1.5G</td>
 </tr>
 <!-- <tr>
    <td>bofenghuang/whisper-large-v3-french-distil-dec2</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
 </tr> -->
 <tr>
    <td>distil-whisper/distil-large-v2</td>
    <td>2.2G</td>
    <td>3.2G</td>
    <td>4.8G</td>
    <td>4.4G</td>
 </tr>
 <tr>
    <td>large (large-v3, ...)</td>
    <td>2.8G</td>
    <td>4.8G</td>
    <td>8.2G</td>
    <td>10.4G</td>
 </tr>
 <tr>
    <td>large-v3-turbo</td>
    <td>1.3G</td>
    <td>2.0G</td>
    <td>4.0G</td>
    <td>6.0G</td>
 </tr>
</table>

### Model(s)

LinTO-STT-Whisper works with a Whisper model to perform Automatic Speech Recognition.
If not downloaded already, the model will be downloaded when calling the first transcription,
and can occupy several GB of disk space.

#### Optional alignment model (deprecated)

LinTO-STT-Whisper has also the option to work with a wav2vec model to perform word alignment.
The wav2vec model can be specified either
* (TorchAudio) with a string corresponding to a `torchaudio` pipeline (e.g. `WAV2VEC2_ASR_BASE_960H`) or
* (HuggingFace's Transformers) with a string corresponding to a HuggingFace repository of a wav2vec model (e.g. `jonatasgrosman/wav2vec2-large-xlsr-53-english`), or
* (SpeechBrain) with a path corresponding to a folder with a SpeechBrain model

Default wav2vec models are provided for French (fr), English (en), Spanish (es), German (de), Dutch (nl), Japanese (ja), Chinese (zh).

But we advise not to use a companion wav2vec alignment model.
This is not needed neither tested anymore.


### (micro-service) Service broker and shared folder
The STT only entry point in task mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, STT-Worker use a shared storage folder (SHARED_FOLDER).

## Deploy LinTO-STT-Whisper

### 1- First step is to build or pull the image

```bash
git clone https://github.com/linto-ai/linto-stt.git
cd linto-stt
docker build . -f whisper/Dockerfile.ctranslate2 -t linto-stt-whisper:latest
```
or

```bash
docker pull lintoai/linto-stt-whisper
```

### 2- Fill the .env file

An example of .env file is provided in [whisper/.envdefault](https://github.com/linto-ai/linto-stt/blob/master/whisper/.envdefault).

| PARAMETER | DESCRIPTION | EXEMPLE |
|---|---|---|
| SERVICE_MODE | (Required) STT serving mode see [Serving mode](#serving-mode) | `http` \| `task` \| `websocket` |
| MODEL | (Required) Path to a Whisper model, type of Whisper model used, or HuggingFace identifier of a Whisper model. | `large-v3` \| `distil-whisper/distil-large-v2` \| \<ASR_PATH\> \| ... |
| LANGUAGE | Language to recognize | `*` \| `fr` \| `fr-FR` \| `French` \| `en` \| `en-US` \| `English` \| ... |
| PROMPT | Prompt to use for the Whisper model | `some free text to encourage a certain transcription style (disfluencies, no punctuation, ...)` |
| DEVICE | Device to use for the model (by default, GPU/CUDA is used if it is available, CPU otherwise) | `cpu` \| `cuda` |
| NUM_THREADS | Number of threads (maximum) to use for things running on CPU | `1` \| `4` \| ... |
| CUDA_VISIBLE_DEVICES | GPU device index to use, when running on GPU/CUDA. We also recommend to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU machines | `0` \| `1` \| `2` \| ... |
| CONCURRENCY | Maximum number of parallel requests (number of workers minus one) | `2` |
| VAD | Voice Activity Detection method. Use "false" to disable. If not specified, the default is auditok VAD. | `true` \| `false` \| `1` \| `0` \| `auditok` \| `silero`
| VAD_DILATATION | How much (in sec) to enlarge each speech segment detected by the VAD. If not specified, the default is auditok 0.5 | `0.1` \| `0.5` \| ...
| VAD_MIN_SPEECH_DURATION | Minimum duration (in sec) of a speech segment. If not specified, the default is 0.1 | `0.1` \| `0.5` \| ...
| VAD_MIN_SILENCE_DURATION | Minimum duration (in sec) of a silence segment. If not specified, the default is 0.1 | `0.1` \| `0.5` \| ...
| ENABLE_STREAMING | (For the http mode) Legacy, it redirects to websocket mode if enabled | `true` \| `false` |
| USE_ACCURATE | Use more expensive parameters for better transcriptions (but slower). If not specified, the default is true |  `true` \| `false` \| `1` \| `0` |
| STREAMING_PORT | (For the websocket mode) the listening port for ingoing WS connexions. If not specified, the default is 80 | `80` |
| STREAMING_MIN_CHUNK_SIZE | The minimal size of the buffer (in seconds) before transcribing. If not specified, the default is 0.5 | `0.5` \| `26` \| ... |
| STREAMING_BUFFER_TRIMMING_SEC | The maximum targeted length of the buffer (in seconds). It tries to cut after a transcription has been made. If not specified, the default is 8 | `8` \| `10` \| ... |
| STREAMING_PAUSE_FOR_FINAL | The minimum duration of silence (in seconds) needed to be able to output a final. If not specified, the default is 1.5 | `0.5` \| `2` \| ... |
| STREAMING_TIMEOUT_FOR_SILENCE | If VAD is applied locally before sending data to the server, this will allow the server to find the silence. The `packet duration` is determined from the first packet. If a packet is not received during `packet duration * STREAMING_TIMEOUT_FOR_SILENCE` it considers that a silence (lasting the packet duration) is present. Value should be between 1 and 2. If not specified, the default is 1.5 | `1.8` \| ... |
| SERVICE_NAME | (For the task mode only) queue's name for task processing | `my-stt` |
| SERVICE_BROKER | (For the task mode only) URL of the message broker | `redis://my-broker:6379` |
| BROKER_PASS | (For the task mode only) broker password | `my-password` \| (empty) |
| ALIGNMENT_MODEL | (Deprecated) Path to the wav2vec model for word alignment, or name of HuggingFace repository or torchaudio pipeline | `WAV2VEC2_ASR_BASE_960H` \| `jonatasgrosman/wav2vec2-large-xlsr-53-english` \| \<WAV2VEC_PATH\> \| ... |


#### MODEL environment variable

**Warning:**
The model will be (downloaded if required and) loaded in memory when calling the first transcription.
When using a Whisper model from Hugging Face (transformers) along with ctranslate2 (faster_whisper),
it will also download torch library to make the conversion from torch to ctranslate2.

If you want to preload the model (and later specify a path `<ASR_PATH>` as `MODEL`),
you may want to download one of OpenAI Whisper models:
* Mutli-lingual Whisper models can be downloaded with the following links:
    * [tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
    * [base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)
    * [small](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt)
    * [medium](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
    * [large-v1](https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt)
    * [large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)
    * [large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)
    * [large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt)
* Whisper models specialized for English can also be found here:
    * [tiny.en](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt)
    * [base.en](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt)
    * [small.en](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt)
    * [medium.en](https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt)

If you already used Whisper in the past locally using [OpenAI-Whipser](https://github.com/openai/whisper), models can be found under `~/.cache/whisper`.

The same apply for Whisper models from Hugging Face (transformers), as for instance https://huggingface.co/distil-whisper/distil-large-v2
(you can either download the model or use the Hugging Face identifier `distil-whisper/distil-large-v2`).

#### LANGUAGE

The `LANGUAGE` environment variable
can be used to set the default language
(which can be "`*`" for automatic language detection).
Note that the `language` can also be passed as a parameter in the request: in this case, it will override the `LANGUAGE` environment variable.

The language value can be:
* the wildcard "`*`", for automatic language detection (by Whisper model),
* a language BCP-47 code ("`fr-FR`", "`en-US`", "`yue-HK`", ...),
* a language code of two or three letters  ("`fr`", "`en`", "`yue`", ...).
  Note that this is the only part of the BCP-47 code that is effectively used.
* a language name ("`French`", "`English`", "`Cantonese`", ...).

The list of languages supported by Whisper are:
`af`(afrikaans), `am`(amharic), `ar`(arabic), `as`(assamese), `az`(azerbaijani),
`ba`(bashkir), `be`(belarusian), `bg`(bulgarian), `bn`(bengali), `bo`(tibetan), `br`(breton), `bs`(bosnian),
`ca`(catalan), `cs`(czech), `cy`(welsh), `da`(danish), `de`(german), `el`(greek), `en`(english), `es`(spanish),
`et`(estonian), `eu`(basque), `fa`(persian), `fi`(finnish), `fo`(faroese), `fr`(french), `gl`(galician),
`gu`(gujarati), `ha`(hausa), `haw`(hawaiian), `he`(hebrew), `hi`(hindi), `hr`(croatian), `ht`(haitian creole),
`hu`(hungarian), `hy`(armenian), `id`(indonesian), `is`(icelandic), `it`(italian), `ja`(japanese),
`jw`(javanese), `ka`(georgian), `kk`(kazakh), `km`(khmer), `kn`(kannada), `ko`(korean), `la`(latin),
`lb`(luxembourgish), `ln`(lingala), `lo`(lao), `lt`(lithuanian), `lv`(latvian), `mg`(malagasy), `mi`(maori),
`mk`(macedonian), `ml`(malayalam), `mn`(mongolian), `mr`(marathi), `ms`(malay), `mt`(maltese), `my`(myanmar),
`ne`(nepali), `nl`(dutch), `nn`(nynorsk), `no`(norwegian), `oc`(occitan), `pa`(punjabi), `pl`(polish),
`ps`(pashto), `pt`(portuguese), `ro`(romanian), `ru`(russian), `sa`(sanskrit), `sd`(sindhi), `si`(sinhala),
`sk`(slovak), `sl`(slovenian), `sn`(shona), `so`(somali), `sq`(albanian), `sr`(serbian), `su`(sundanese),
`sv`(swedish), `sw`(swahili), `ta`(tamil), `te`(telugu), `tg`(tajik), `th`(thai), `tk`(turkmen), `tl`(tagalog),
`tr`(turkish), `tt`(tatar), `uk`(ukrainian), `ur`(urdu), `uz`(uzbek), `vi`(vietnamese), `yi`(yiddish),
`yo`(yoruba), `zh`(chinese).

Model `large-v3` and recent models derived from it also supports `yue`(cantonese).

#### SERVING_MODE
![Serving Modes](https://i.ibb.co/qrtv3Z6/platform-stt.png)

STT can be used in three ways:
* Through an [HTTP API](#http-server) using the **http**'s mode.
* Through a [message broker](#celery-task) using the **task**'s mode.
* Through a [websocket server](#websocket-server) using the **websocket**'s mode.

Mode is specified using the .env value or environment variable ```SERVING_MODE```.
```bash
SERVICE_MODE=http
```

### HTTP Server
The HTTP serving mode deploys a HTTP server and a swagger-ui to allow transcription request on a dedicated route.

The SERVICE_MODE value in the .env should be set to ```http```.

```bash
docker run --rm \
-p HOST_SERVING_PORT:80 \
--env-file .env \
linto-stt-whisper:latest
```

This will run a container providing an [HTTP API](#http-api) binded on the host HOST_SERVING_PORT port.

You may also want to add specific options:
* To enable GPU capabilities, add ```--gpus all```.
  Note that you can use environment variable `DEVICE=cuda` to make sure GPU is used (and maybe set `CUDA_VISIBLE_DEVICES` if there are several available GPU cards).
* To mount a local cache folder `<CACHE_PATH>` (e.g. "`$HOME/.cache`") and avoid downloading models each time,
  use ```-v <CACHE_PATH>:/root/.cache```
  If you use `MODEL=/opt/model.pt` environment variable, you may want to mount the model file (or folder) with option ```-v <ASR_PATH>:/opt/model.pt```.
* If you want to specifiy a custom alignment model already downloaded in a folder `<WAV2VEC_PATH>`,
  you can add option ```-v <WAV2VEC_PATH>:/opt/wav2vec``` and environment variable ```ALIGNMENT_MODEL=/opt/wav2vec```.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `HOST_SERVING_PORT` | Host serving port | 8080 |
| `<CACHE_PATH>` | Path to a folder to download wav2vec alignment models when relevant | /home/username/.cache |
| `<ASR_PATH>` | Path to the Whisper model on the host machine mounted to /opt/model.pt | /my/path/to/models/medium.pt |
| `<WAV2VEC_PATH>` | Path to a folder to a custom wav2vec alignment model |  /my/path/to/models/wav2vec |

### Celery task
The TASK serving mode connect a celery worker to a message broker.

The SERVICE_MODE value in the .env should be set to ```task```.

You need a message broker up and running at MY_SERVICE_BROKER.

```bash
docker run --rm \
-v SHARED_AUDIO_FOLDER:/opt/audio \
--env-file .env \
linto-stt-whisper:latest
```

You may also want to add specific options:
* To enable GPU capabilities, add ```--gpus all```.
  Note that you can use environment variable `DEVICE=cuda` to make sure GPU is used (and maybe set `CUDA_VISIBLE_DEVICES` if there are several available GPU cards).
* To mount a local cache folder `<CACHE_PATH>` (e.g. "`$HOME/.cache`") and avoid downloading models each time,
  use ```-v <CACHE_PATH>:/root/.cache```
  If you use `MODEL=/opt/model.pt` environment variable, you may want to mount the model file (or folder) with option ```-v <ASR_PATH>:/opt/model.pt```.
* If you want to specifiy a custom alignment model already downloaded in a folder `<WAV2VEC_PATH>`,
  you can add option ```-v <WAV2VEC_PATH>:/opt/wav2vec``` and environment variable ```ALIGNMENT_MODEL=/opt/wav2vec```.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `<SHARED_AUDIO_FOLDER>` | Shared audio folder mounted to /opt/audio | /my/path/to/models/vosk-model |
| `<CACHE_PATH>` | Path to a folder to download wav2vec alignment models when relevant | /home/username/.cache |
| `<ASR_PATH>` | Path to the Whisper model on the host machine mounted to /opt/model.pt | /my/path/to/models/medium.pt |
| `<WAV2VEC_PATH>` | Path to a folder to a custom wav2vec alignment model |  /my/path/to/models/wav2vec |

### Websocket Server
Websocket server's mode deploy a streaming transcription service only. 

The SERVICE_MODE value in the .env should be set to ```websocket```.

<!-- Usage is the same as the [http streaming API](#streaming). -->

<!-- The /streaming route is accessible if the ENABLE_STREAMING environment variable is set to true. -->

The exchanges are structured as followed:
1. Client send a json {"config": {"sample_rate":16000, "language":"en"}}. Language is optional, if not specified it will use the language from the env.
2. Client send audio chunk (go to 3- ) or {"eof" : 1} (go to 5-).
3. Server send either a partial result {"partial" : "this is a "} or a final result {"text": "this is a transcription"}.
4. Back to 2-
5. Server send a final result and close the connexion.


We advise to run streaming on a GPU device.

How to choose the 2 streaming parameters "`STREAMING_MIN_CHUNK_SIZE`" and "`STREAMING_BUFFER_TRIMMING_SEC`"?
- If you want a low latency (2 to a 5 seconds on a NVIDIA 4090 Laptop), choose a small value for "STREAMING_MIN_CHUNK_SIZE" like 0.5 seconds (to avoid making useless predictions).
For "`STREAMING_BUFFER_TRIMMING_SEC`", around 10 seconds is a good compromise between keeping latency low and having a good transcription accuracy.
Depending on the hardware and the model, this value should go from 6 to 15 seconds.
- If you can efford to have a high latency (30 seconds) and want to minimize GPU activity, choose a big value for "`STREAMING_MIN_CHUNK_SIZE`", such as 26s (which will give latency around 30 seconds).
For "`STREAMING_BUFFER_TRIMMING_SEC`", you will need to have a value lower than "`STREAMING_MIN_CHUNK_SIZE`".
Good results can be obtained by using a value between 6 and 12 seconds.
The lower the value, the lower the GPU usage will be, but you will probably degrade transcription accuracy (more error on words because the model will miss some context).

<!-- Concerning transcription accuracies, some tests on transcription in French gave the following results:
* around 20% WER (Word Error Rate) with offline transcription,
* around 30% WER with high latency streaming (around 30 seconds latency on a GPU), and
* around 40% WER with low latency streaming (beween 2 and 3 seconds latency on average on a GPU). -->

## Usages
### HTTP API
#### /healthcheck
Returns the state of the API

Method: GET

Returns "1" if healthcheck passes.

#### /transcribe
Transcription API

* Method: POST
* Response content: text/plain or application/json
* File: An Wave file 16b 16Khz
* Language (optional): Override environment variable `LANGUAGE`

Return the transcripted text using "text/plain" or a json object when using "application/json" structure as followed:
```json
{
    "text" : "This is the transcription as text",
    "words": [
        {
        "word" : "This",
        "start": 0.0,
        "end": 0.124,
        "conf": 0.82341
        },
        ...
    ],
    "language": "en",
    "confidence-score": 0.879
}
```

<!-- #### /streaming -->

#### /docs
The /docs route offers a OpenAPI/swagger interface.

### Through the message broker

STT-Worker accepts requests with the following arguments:
```file_path: str, with_metadata: bool```

* <ins>file_path</ins>: Is the location of the file within the shared_folder. /.../SHARED_FOLDER/{file_path}
* <ins>with_metadata</ins>: If True, words timestamps and confidence will be computed and returned. If false, the fields will be empty.

#### Return format
On a successfull transcription the returned object is a json object structured as follow:
```json
{
    "text" : "This is the transcription as text",
    "words": [
        {
        "word" : "This",
        "start": 0.0,
        "end": 0.124,
        "conf": 0.82341
        },
        ...
    ],
    "confidence-score": 0.879
}
```

* The <ins>text</ins> field contains the raw transcription.
* The <ins>word</ins> field contains each word with their time stamp and individual confidence. (Empty if with_metadata=False)
* The <ins>confidence</ins> field contains the overall confidence for the transcription. (0.0 if with_metadata=False)


## Tests

### Curl
You can test your http API using curl:

```bash 
curl -X POST "http://YOUR_SERVICE:YOUR_PORT/transcribe" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@YOUR_FILE;type=audio/x-wav"
```

### Streaming
You can test your streaming API using a websocket:

```bash 
python test/test_streaming.py --server ws://YOUR_SERVICE:YOUR_PORT/streaming --audio_file test/bonjour.wav
```

## License
This project is developped under the AGPLv3 License (see LICENSE).

## Acknowlegment.

* [Ctranslate2](https://github.com/OpenNMT/CTranslate2)
   * [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* [OpenAI Whisper](https://github.com/openai/whisper)
   * [Whisper-Timestamped](https://github.com/linto-ai/whisper-timestamped)
* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [TorchAudio](https://github.com/pytorch/audio)
* [Whisper_Streaming](https://github.com/ufal/whisper_streaming)
