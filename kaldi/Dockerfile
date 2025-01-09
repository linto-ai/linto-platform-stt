FROM python:3.9
LABEL maintainer="contact@linto.ai, jlouradour@linagora.com, dgaynullin@linagora.com"

ARG KALDI_MKL

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        nano \
        bzip2 \
        unzip \
        xz-utils \
        sox \
        g++ \
        make \
        cmake \
        curl \
        netcat-traditional \
        git \
        zlib1g-dev \
        automake \
        autoconf \
        libtool \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build vosk-kaldi
RUN git clone -b vosk --single-branch https://github.com/alphacep/kaldi /opt/kaldi \
    && cd /opt/kaldi/tools \
    && sed -i 's:status=0:exit 0:g' extras/check_dependencies.sh \
    && sed -i 's:--enable-ngram-fsts:--enable-ngram-fsts --disable-bin:g' Makefile \
    && make -j $(( $(nproc) < 8 ? $(nproc) : 8 )) openfst cub \
    && if [ "x$KALDI_MKL" != "x1" ] ; then \
          extras/install_openblas_clapack.sh; \
       else \
          extras/install_mkl.sh; \
       fi \
    && cd /opt/kaldi/src \
    && if [ "x$KALDI_MKL" != "x1" ] ; then \
          ./configure --mathlib=OPENBLAS_CLAPACK --shared; \
       else \
          ./configure --mathlib=MKL --shared; \
       fi \
    && sed -i 's:-msse -msse2:-msse -msse2:g' kaldi.mk \
    && sed -i 's: -O1 : -O3 :g' kaldi.mk \
    && make -j $(( $(nproc) < 8 ? $(nproc) : 8 )) online2 lm rnnlm

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install python dependencies
COPY kaldi/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Custom Vosk API
RUN git clone --depth 1 https://github.com/alphacep/vosk-api /opt/vosk-api && cd /opt/vosk-api/python && \
    cd /opt/vosk-api/src \
    && KALDI_MKL=$KALDI_MKL KALDI_ROOT=/opt/kaldi make -j $(( $(nproc) < 8 ? $(nproc) : 8 )) \
    && cd /opt/vosk-api/python \
    && python3 ./setup.py install

WORKDIR /usr/src/app

# Install what's needed for punctuation
COPY punctuation/requirements.cpu.txt ./
RUN pip install --no-cache-dir -r requirements.cpu.txt -f https://download.pytorch.org/whl/torch_stable.html
RUN rm requirements.cpu.txt

# Copy code
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY websocket /usr/src/app/websocket
COPY document /usr/src/app/document
COPY kaldi/stt /usr/src/app/stt
COPY kaldi/docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./
COPY kaldi/lin_to_vosk.py /usr/src/app/lin_to_vosk.py
COPY punctuation ./punctuation

RUN mkdir -p /var/log/supervisor/

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/stt"

HEALTHCHECK CMD ./healthcheck.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
