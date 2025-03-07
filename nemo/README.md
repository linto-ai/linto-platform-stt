# LinTO-STT-NeMo

LinTO-STT-NeMo is an API for Automatic Speech Recognition (ASR) based on [NeMo toolkit](https://github.com/NVIDIA/NeMo).

LinTO-STT-NeMo can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector. 

It can be used to do offline or real-time transcriptions.

## Pre-requisites

### Requirements

The transcription service requires [docker](https://www.docker.com/products/docker-desktop/) up and running.

For GPU capabilities, it is also needed to install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Hardware

To run the transcription models you'll need:
* At least 15GB of disk space to build the docker image
  and models can occupy several GB of disk space depending on the model size (it can be up to 5GB).
* Up to 8GB of RAM depending on the model used.
* One CPU per worker. Inference time scales on CPU performances.

### Model(s)

LinTO-STT-NeMo works with a NeMo compatible ASR model to perform Automatic Speech Recognition.
If not downloaded already, the model will be downloaded when calling the first transcription, and can occupy several GB of disk space.

### (micro-service) Service broker and shared folder
The STT only entry point in task mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, STT-Worker use a shared storage folder (SHARED_FOLDER).

## Deploy LinTO-STT-Whisper

### 1- First step is to build or pull the image

```bash
git clone https://github.com/linto-ai/linto-stt.git
cd linto-stt
docker build . -f nemo/Dockerfile -t linto-stt-nemo:latest
```

### 2- Fill the .env file

An example of .env file is provided in [nemo/.envdefault](https://github.com/linto-ai/linto-stt/blob/master/nemo/.envdefault).

| PARAMETER | DESCRIPTION | EXEMPLE |
|---|---|---|
| SERVICE_MODE | (Required) STT serving mode see [Serving mode](#serving-mode) | `http` \| `task` \| `websocket` |
| MODEL | (Required) Path to a NeMo model or HuggingFace identifier. | `nvidia/parakeet-ctc-1.1b` \| `nvidia/stt_fr_fastconformer_hybrid_large_pc` \| \<ASR_PATH\> \| ... |
| ARCHITECTURE | (Required) The architecture of the model used. Suported (and tested) architectures are CTC, Hybrid and RNNT models. | `hybrid_bpe` \| `ctc_bpe` \| `rnnt_bpe` \|\<ASR_PATH\> \| ... |
| DEVICE | Device to use for the model (by default, GPU/CUDA is used if it is available, CPU otherwise) | `cpu` \| `cuda` |
| NUM_THREADS | Number of threads (maximum) to use for things running on CPU | `1` \| `4` \| ... |
| CUDA_VISIBLE_DEVICES | GPU device index to use, when running on GPU/CUDA. We also recommend to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU machines | `0` \| `1` \| `2` \| ... |
| CONCURRENCY | Maximum number of parallel requests (number of workers minus one) | `2` |
| LONG_FILE_THRESHOLD | For offline decoding, a file longer than that will be split into smaller parts (long form transcription). This value depends on your VRAM/RAM amount. Default is 480s (8mins) | `480` \| `240`
| LONG_FILE_CHUNK_LEN | For long form transcription, size of the parts into which the audio is splitted. This value depends on your VRAM/RAM amount. Default is 240s (4mins) | `240` \| `120`
| LONG_FILE_CHUNK_CONTEXT_LEN | For long form transcription, size of the context added at the begining and end of each chunk. Default is 10s (10s before and after the chunk) | `10` \| `5`
| VAD | Voice Activity Detection method. Use "false" to disable. If not specified, the default is auditok VAD. | `true` \| `false` \| `1` \| `0` \| `auditok` \| `silero`
| VAD_DILATATION | How much (in sec) to enlarge each speech segment detected by the VAD. If not specified, the default is auditok 0.5 | `0.1` \| `0.5` \| ...
| VAD_MIN_SPEECH_DURATION | Minimum duration (in sec) of a speech segment. If not specified, the default is 0.1 | `0.1` \| `0.5` \| ...
| VAD_MIN_SILENCE_DURATION | Minimum duration (in sec) of a silence segment. If not specified, the default is 0.1 | `0.1` \| `0.5` \| ...
| ENABLE_STREAMING | (For the http mode) Legacy, it redirects to websocket mode if enabled | `true` \| `false` |
| STREAMING_PORT | (For the websocket mode) the listening port for ingoing WS connexions. If not specified, the default is 80 | `80` |
| STREAMING_MIN_CHUNK_SIZE | The minimal size of the buffer (in seconds) before transcribing. If not specified, the default is 0.5 | `0.5` \| `26` \| ... |
| STREAMING_BUFFER_TRIMMING_SEC | The maximum targeted length of the buffer (in seconds). It tries to cut after a transcription has been made. If not specified, the default is 8 | `8` \| `10` \| ... |
| STREAMING_PAUSE_FOR_FINAL | The minimum duration of silence (in seconds) needed to be able to output a final. If not specified, the default is 1.5 | `0.5` \| `2` \| ... |
| STREAMING_TIMEOUT_FOR_SILENCE | If VAD is applied locally before sending data to the server, this will allow the server to find the silence. The `packet duration` is determined from the first packet. If a packet is not received during `packet duration * STREAMING_TIMEOUT_FOR_SILENCE` it considers that a silence (lasting the packet duration) is present. Value should be between 1 and 2. If not specified, the default is 1.5 | `1.8` \| ... |
| SERVICE_NAME | (For the task mode only) queue's name for task processing | `my-stt` |
| SERVICE_BROKER | (For the task mode only) URL of the message broker | `redis://my-broker:6379` |
| BROKER_PASS | (For the task mode only) broker password | `my-password` \| (empty) |


#### MODEL environment variable

**Warning:**
The model will be (downloaded if required and) loaded in memory when calling the first transcription.

If you want to preload the model (and later specify a path `<ASR_PATH>` as `MODEL`),
you may want to download one of NeMo models:
   * [French large fast conformer with punctuations made by NVIDIA](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc)
   * [French large fast conformer made by Bofeng Huang](https://huggingface.co/bofenghuang/stt_fr_fastconformer_hybrid_large)
   * [English large fast conformer made by NVIDIA](https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large)
   * [English XXL fast conformer made by NVIDIA](https://huggingface.co/nvidia/parakeet-ctc-1.1b)
   * Soon models trained by LINAGORA
   * More stt models are available in [NVIDIA](https://huggingface.co/nvidia) huggingface

NeMo models from Hugging Face (transformers), as for instance https://huggingface.co/nvidia/parakeet-ctc-1.1b (you can either download the model or use the Hugging Face identifier `nvidia/parakeet-ctc-1.1b`).

#### LANGUAGE

Tested models are not multi-lingual, so, the `LANGUAGE` is environment variable is just as information and does not have any impacts. By default, it will try to retrieve the language from the model name.
Note that the `language` can also be passed as a parameter in the request: in this case, it will override the `LANGUAGE` environment variable. 

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
linto-stt-nemo:latest
```

This will run a container providing an [HTTP API](#http-api) binded on the host HOST_SERVING_PORT port.

You may also want to add specific options:
* To enable GPU capabilities, add ```--gpus all```.
  Note that you can use environment variable `DEVICE=cuda` to make sure GPU is used (and maybe set `CUDA_VISIBLE_DEVICES` if there are several available GPU cards).
* To mount a local cache folder `<CACHE_PATH>` (e.g. "`$HOME/.cache`") and avoid downloading models each time,
  use ```-v <CACHE_PATH>:/root/.cache```
  If you use `MODEL=/opt/model.nemo` environment variable, you may want to mount the model file (or folder) with option ```-v <ASR_PATH>:/opt/model.nemo```.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `HOST_SERVING_PORT` | Host serving port | 8080 |
| `<CACHE_PATH>` | Path to a folder to download wav2vec alignment models when relevant | /home/username/.cache |
| `<ASR_PATH>` | Path to the NeMo model on the host machine mounted to /opt/model.nemo | /my/path/to/models/stt_fr.nemo |

### Celery task
The TASK serving mode connect a celery worker to a message broker.

The SERVICE_MODE value in the .env should be set to ```task```.

You need a message broker up and running at MY_SERVICE_BROKER.

```bash
docker run --rm \
-v SHARED_AUDIO_FOLDER:/opt/audio \
--env-file .env \
linto-stt-nemo:latest
```

You may also want to add specific options:
* To enable GPU capabilities, add ```--gpus all```.
  Note that you can use environment variable `DEVICE=cuda` to make sure GPU is used (and maybe set `CUDA_VISIBLE_DEVICES` if there are several available GPU cards).
* To mount a local cache folder `<CACHE_PATH>` (e.g. "`$HOME/.cache`") and avoid downloading models each time,
  use ```-v <CACHE_PATH>:/root/.cache```
  If you use `MODEL=/opt/model.nemo` environment variable, you may want to mount the model file (or folder) with option ```-v <ASR_PATH>:/opt/model.nemo```.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `<SHARED_AUDIO_FOLDER>` | Shared audio folder mounted to /opt/audio | /my/path/to/models/vosk-model |
| `<CACHE_PATH>` | Path to a folder to download wav2vec alignment models when relevant | /home/username/.cache |
| `<ASR_PATH>` | Path to the NeMo model on the host machine mounted to /opt/model.nemo | /my/path/to/models/stt_fr.nemo |

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

 We recommend to use a VAD on the server side (silero for example).

How to choose the 2 streaming parameters `STREAMING_MIN_CHUNK_SIZE` and `STREAMING_BUFFER_TRIMMING_SEC`?
- If you want a low latency (2 to a 5 seconds on a NVIDIA 4090 Laptop), choose a small value for "STREAMING_MIN_CHUNK_SIZE" like 0.5 seconds (to avoid making useless predictions).
For `STREAMING_BUFFER_TRIMMING_SEC`, around 10 seconds is a good compromise between keeping latency low and having a good transcription accuracy.
Depending on the hardware and the model, this value should go from 6 to 15 seconds.
- If you can efford to have a high latency (30 seconds) and want to minimize GPU activity, choose a big value for `STREAMING_MIN_CHUNK_SIZE`, such as 26s (which will give latency around 30 seconds).
For `STREAMING_BUFFER_TRIMMING_SEC`, you will need to have a value lower than `STREAMING_MIN_CHUNK_SIZE`.
Good results can be obtained by using a value between 6 and 12 seconds.
The lower the value, the lower the GPU usage will be (because audio buffer will be smaller), but you will probably degrade transcription accuracy (more error on words because the model will miss some context).

The `STREAMING_PAUSE_FOR_FINAL` value will depend on your type of speech. On prepared speech for example, you can probably lower it whereas on real discussions you can leave it as default or increase it. 

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

* [NeMo](https://github.com/NVIDIA/NeMo)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [TorchAudio](https://github.com/pytorch/audio)
* [Whisper_Streaming](https://github.com/ufal/whisper_streaming)
