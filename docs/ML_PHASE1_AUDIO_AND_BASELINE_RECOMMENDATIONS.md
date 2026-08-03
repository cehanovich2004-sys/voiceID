# VoiceID ML-рекомендации: аудиовалидация и baseline

Аудитория: CTO, Lead Software Engineer
Контекст: Phase 1, подготовка к Phase 2-7
Статус документа: рекомендации и ML guardrails, не ADR
Задача: speaker verification one-to-one, не speaker identification, не speech
recognition, не VAD, не diarization и не anti-spoofing.

## 1. Итог

ML status: `COMMENTS`

VoiceID MVP следует формулировать как прототип text-independent speaker
verification:

> Даны две аудиозаписи. Нужно оценить, принадлежат ли они одному и тому же
> говорящему.

Система не должна показывать cosine similarity как вероятность совпадения,
пока нет размеченного validation set, определения target/non-target пар,
калибровки score, выбора порога и оценки ошибок.

CTO review зафиксировал границы Phase 2:

- Phase 2 реализует только техническую загрузку и валидацию WAV;
- Phase 2 не выполняет downmix, resampling, normalization, VAD, denoising,
  silence trimming, embedding extraction или scoring;
- Phase 2 не определяет пригодность записи для speaker verification;
- реальные аудио и speaker embeddings нельзя коммитить в Git.

SpeechBrain ECAPA-TDNN остается предварительным кандидатом для Phase 4.
Окончательный выбор baseline-модели должен быть оформлен отдельным ADR.

## 2. Корректность ML-задачи

MVP-задача: speaker verification one-to-one.

Вход:

- аудиозапись A;
- аудиозапись B.

Выход до evaluation и threshold selection:

- технический результат валидации аудио;
- позднее, не раньше фазы подключения baseline: raw cosine score.

Запрещено до evaluation и threshold selection:

- `MATCH`;
- `NO_MATCH`;
- `UNCERTAIN`;
- probability-like output;
- claims production-grade biometric accuracy.

Положительный класс в будущих экспериментах:

- обе записи содержат одного и того же целевого говорящего.

Отрицательный класс в будущих экспериментах:

- записи содержат разных говорящих.

Нельзя смешивать с другими задачами:

- speaker identification: выбор личности из базы;
- speech recognition: распознавание текста;
- VAD: поиск speech/non-speech фрагментов;
- diarization: определение "кто когда говорил";
- anti-spoofing: обнаружение replay, TTS, voice conversion, deepfake.

Эти задачи могут понадобиться позже, но они не являются целью Phase 2.

## 3. Phase 2: аудиозагрузка и техническая валидация

Phase 2 должна реализовать консервативный слой audio validation без model
inference. Результат валидации должен быть структурированным: metadata,
warnings и errors.

Техническая валидность файла не гарантирует, что запись пригодна для speaker
verification. Она означает только, что файл соответствует входному техническому
контракту MVP.

### 3.1 Утвержденный входной стандарт Phase 2

Phase 2 принимает файл только если он соответствует всем требованиям:

- container: RIFF/WAVE;
- codec: PCM 16-bit;
- channel count: mono или stereo;
- sample rate: 8 kHz, 16 kHz, 22.05 kHz, 44.1 kHz или 48 kHz;
- duration: от 1 до 60 секунд включительно;
- maximum duration default: 60 seconds;
- maximum duration must be configurable.

Hard errors:

- path does not exist;
- file is unreadable;
- file is empty;
- container is not RIFF/WAVE;
- unsupported codec inside WAV, including compressed WAV;
- WAV is not PCM 16-bit;
- decoded audio has zero samples;
- duration is below 1 second;
- duration is above configured maximum duration;
- sample rate is missing or not in the allowed list;
- channel count is not mono or stereo;
- all decoded samples are zero or near-zero.

Warnings:

- stereo input;
- sample rate is not 16 kHz.

Phase 2 does not perform:

- stereo-to-mono downmix;
- resampling to 16 kHz;
- loudness or peak normalization;
- silence trimming;
- VAD;
- speaker counting;
- noise floor classification;
- replay or re-recording detection;
- actual speech duration estimation;
- speaker-verification suitability decision.

### 3.2 Validation statuses

Validation result must use exactly these public statuses:

- `VALID`: file satisfies the technical Phase 2 contract without warnings;
- `VALID_WITH_WARNINGS`: file satisfies the technical contract, but has
  non-blocking warnings such as stereo input or sample rate not equal to
  16 kHz;
- `INVALID`: file violates the technical contract and must not continue through
  the MVP audio pipeline.

`VALID` and `VALID_WITH_WARNINGS` do not mean "valid for biometric use" or
"valid for speaker verification". They mean only "technically accepted by the
Phase 2 loader".

### 3.3 Public validation result contract

Public API responses, logs and user-facing error messages must not include
internal filesystem paths. A safe filename is acceptable. The internal path may
exist inside trusted application code but must not be exposed.

Recommended public metadata fields:

- safe filename;
- detected container;
- detected codec;
- sample rate;
- channel count;
- duration seconds;
- bit depth;
- total samples;
- peak amplitude;
- RMS level;
- validation status;
- validation warnings;
- validation errors.

Recommended warning/error structure:

- stable machine-readable code;
- short user-safe message;
- field name when applicable;
- measured value when safe to expose;
- expected value or allowed range.

Do not rely on file extension alone. The loader must decode the file header and
validate actual audio properties.

### 3.4 Checks explicitly deferred beyond Phase 2

The following checks must not be treated as mandatory Phase 2 functionality:

- detecting multiple speakers;
- estimating high noise floor;
- detecting replay, speaker playback or re-recording;
- estimating actual speech duration;
- deciding whether audio is suitable for speaker verification.

These checks require later components: VAD, diarization, signal-processing
quality metrics, anti-spoofing models, speaker embedding models or manual
dataset labeling. They may become warnings or quality gates in later phases
only after implementation and evaluation.

## 4. Downstream ML audio target

The downstream target for the future ML pipeline is:

- mono;
- 16 kHz;
- floating-point waveform in a consistent range.

This is a future ML preprocessing target, not Phase 2 behavior. Phase 2 should
only report that stereo input or sample rate not equal to 16 kHz will require
future preprocessing.

Normalization, VAD, denoising and silence trimming are deferred. They must not
be enabled by default until experiments show their effect on verification
scores and error rates.

## 5. Storage и privacy

Голосовые записи и speaker embeddings являются biometric data. Для MVP:

- использовать только записи участников с явным согласием или открытые
  датасеты с подходящей лицензией;
- хранить записи вне репозитория;
- не загружать customer audio в public GitHub;
- не коммитить реальные аудио и speaker embeddings в Git;
- документировать, кто имеет доступ к записям;
- определить retention и deletion rules до сбора реальных пользовательских
  аудио;
- не хранить лишние идентифицирующие metadata;
- считать embeddings чувствительными biometric templates, а не безобидными
  числовыми векторами.

## 6. Сравнение baseline-моделей

Не подключать несколько тяжелых speech frameworks одновременно. Сначала нужно
выбрать один baseline, зафиксировать версии и получить воспроизводимые scores.

| Candidate | Fit for MVP | License / access | Strengths | Concerns |
| --- | --- | --- | --- | --- |
| SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | Preliminary Phase 4 candidate | Model card Apache-2.0; SpeechBrain toolkit Apache-2.0 | Простая speaker-verification API, HF artifact около 89 MB, 192-dim embeddings, сильный VoxCeleb result в документации SpeechBrain | Нужен отдельный ADR; нужно pin model revision и dependency versions; checkpoints pickle-based; phone-channel и noisy-domain quality нужно валидировать локально |
| pyannote embedding (`pyannote/embedding`) | Useful candidate, not first choice | Model card MIT, но нужен HF gated access и acceptance conditions | Зрелая audio ecosystem; прямое embedding extraction; model card сообщает 2.8% EER на VoxCeleb 1 при cosine distance без VAD/PLDA | Gated access усложняет CI/reproducibility; model card предлагает рассмотреть pyannoteAI для production; ecosystem больше diarization-oriented |
| NVIDIA NeMo TitaNet-L / ECAPA | Strong technical candidate for later benchmark | NeMo source Apache-2.0; model artifact/license terms нужно проверять отдельно | Сильный speaker-recognition toolkit; TitaNet-L card сообщает 25.3M params, 192-dim embeddings, VoxCeleb clean EER около 0.68%; training включает Fisher/Switchboard | Более тяжелый dependency stack; NVIDIA ecosystem; CPU latency и footprint нужно измерить локально |
| WeSpeaker | Strong research/production candidate for later benchmark | Apache-2.0 | Toolkit сфокусирован на speaker embeddings и verification; CLI и Python usage; есть PLDA/calibration-oriented workflows и современные recipes | Сложнее для узкого MVP; нужно выбрать конкретную pretrained model; install/reproducibility требуют аккуратности |

CTO decision:

- SpeechBrain ECAPA-TDNN является предварительным кандидатом Phase 4;
- окончательный выбор baseline-модели оформляется отдельным ADR;
- NeMo TitaNet-L, WeSpeaker и pyannote могут быть рассмотрены как challenger
  candidates после появления воспроизводимого baseline.

## 7. Scoring policy

Cosine similarity или cosine distance - это raw model score.

Phase 2 не выполняет scoring.

Raw cosine score может появиться не раньше фазы подключения baseline-модели.

До evaluation и threshold selection запрещено:

- показывать `MATCH`, `NO_MATCH` или `UNCERTAIN`;
- показывать probability-like output;
- писать "96.7% match probability";
- писать "вероятность, что это один и тот же человек";
- писать "identity confirmed";
- писать "biometric authentication passed";
- делать любые claims production-grade accuracy.

После подключения baseline допустимо показывать raw cosine score только с
явным предупреждением, что это не вероятность и не production biometric
decision.

Текущую формулировку product spec про "Match probability" нужно считать
продуктовой гипотезой, а не утвержденным MVP output.

## 8. Экспериментальный протокол

Оценивать нужно пары записей, а не отдельные файлы.

### 8.1 Структура dataset

Для каждой записи нужны:

- speaker ID;
- recording ID;
- session ID;
- device;
- channel type;
- phrase/text condition, если известно;
- environment;
- duration;
- language/accent, если релевантно;
- known quality issues;
- consent/license status.

Минимальный internal pilot:

- 20-30 speakers;
- at least 4 recordings per speaker;
- at least 2 sessions или recording conditions per speaker;
- по возможности balanced same-device и cross-device coverage.

Более сильная MVP evaluation:

- 50+ speakers;
- 5+ recordings per speaker;
- explicit noisy, phone, short-speech и cross-device subsets.

### 8.2 Pair construction

Создать:

- target pairs: same speaker, different recordings;
- non-target pairs: different speakers;
- cross-condition target pairs: same speaker across device/session/noise/phrase;
- matched non-target pairs where possible: похожие device, gender presentation,
  language, channel и duration.

Правила:

- если выполняется tuning, split должен быть by speaker;
- не подбирать threshold на final test set;
- фиксировать random seed для sampled non-target pairs;
- не позволять одному speaker с большим числом записей доминировать в метриках;
- всегда указывать число speakers, recordings, target pairs и non-target pairs.

### 8.3 Conditions для отдельного отчета

Отдельно оценивать после появления evaluation pipeline:

- same device;
- different devices;
- quiet room;
- noise;
- phone channel;
- same phrase;
- different phrases;
- short recordings;
- long recordings;
- changed emotion;
- changed speaking tempo;
- illness or fatigue;
- playback/re-recording through speaker;
- multiple voices.

Некоторые условия могут быть stress-test buckets, а не launch-supported modes.
Но их нужно видеть, потому что это типовые источники ошибок.

### 8.4 Метрики

Primary metrics:

- score distributions для target и non-target pairs;
- ROC;
- DET;
- AUC;
- Equal Error Rate;
- FAR и FRR at candidate thresholds;
- quality at selected operating threshold.

Secondary metrics:

- precision/recall только после определения pair sampling scenario и class
  prevalence;
- bootstrap confidence intervals, желательно resampling by speaker;
- calibration metrics только после реализации calibration.

Threshold selection:

- сначала определить business operating point;
- для security-sensitive сценариев FAR обычно важнее FRR;
- для low-friction demo может быть важнее FRR, но FAR все равно нужно
  показывать;
- рассмотреть `UNCERTAIN` band только после evaluation и threshold selection.

### 8.5 Calibration

Probability-like output требует:

- labeled development set;
- held-out test set;
- target/non-target prior assumption;
- calibration method, например logistic calibration / Platt scaling;
- evaluation of calibrated scores на untouched test data;
- drift monitoring plan.

Даже calibrated probabilities условны: они зависят от dataset и deployment
domain. Это не универсальная вероятность "истинной личности".

## 9. Какие решения нельзя принимать без данных

Нельзя решать по интуиции или нескольким вручную выбранным файлам:

- production threshold;
- можно ли показывать probability;
- какая модель "лучшая": SpeechBrain, NeMo, pyannote или WeSpeaker;
- minimum acceptable speech duration for verification quality;
- помогает или вредит VAD;
- включать ли denoising по умолчанию;
- включать ли normalization или silence trimming по умолчанию;
- нужна ли same-phrase verification;
- качество phone-channel support;
- качество cross-device support;
- fairness/bias claims;
- replay/deepfake resistance;
- целевой FAR/FRR operating point;
- enrollment policy: одна запись или несколько;
- можно ли использовать MVP для customer authentication.

## 10. ML-риски проекта

Domain shift:

- pretrained models обычно benchmarked на датасетах вроде VoxCeleb, а не на
  будущем customer channel VoiceID.

Spoofing:

- speaker verification сам по себе не детектирует replay, TTS, voice conversion
  или deepfake.

Bias:

- error rates могут отличаться по language, accent, gender presentation, age,
  device и recording environment.

Phone channel:

- narrowband speech, compression, packet loss и telephony noise могут сдвигать
  score distributions.

Short speech:

- короткие клипы дают нестабильные embeddings и могут увеличивать false accept
  или false reject.

Quality:

- clipping, music, noise, overlapping speakers и long silence могут создавать
  misleading scores.

Privacy and legal:

- voice recordings и embeddings требуют consent, purpose limitation, access
  control, retention policy и deletion process.

Licensing:

- toolkit license, model-weight license, dataset license и commercial-use rights
  нужно проверять отдельно.

Calibration drift:

- thresholds и calibration могут деградировать при смене devices, users,
  language, microphones или channels.

Data leakage:

- нарезка одной исходной записи на train/dev/test может завысить качество.

Implementation mismatch:

- inconsistent sample rate, stereo handling, normalization или silence trimming
  могут менять scores и маскироваться под поведение модели.

## 11. Рекомендации для следующих фаз

Phase 2:

- реализовать RIFF/WAVE PCM 16-bit loading и metadata validation;
- принимать mono/stereo и утвержденные sample rates;
- enforce duration от 1 до configurable 60 seconds включительно;
- возвращать `VALID`, `VALID_WITH_WARNINGS` или `INVALID`;
- warning для stereo input;
- warning для sample rate not equal to 16 kHz;
- hard error для unsupported codec внутри WAV;
- не выполнять downmix, resampling, normalization, VAD, denoising, silence
  trimming или scoring;
- не определять multiple speakers, high noise floor, replay/re-recording,
  actual speech duration или suitability for speaker verification;
- не возвращать internal filesystem path в API response, логи или
  пользовательские сообщения;
- test audio делать synthetic/generated или хранить только разрешенные локальные
  samples вне public repo.

Phase 3+:

- выбрать и описать mono conversion;
- выбрать target resampling method to 16 kHz;
- отдельно оценить normalization, silence trimming, VAD и denoising до
  включения по умолчанию.

Phase 4:

- создать ADR с выбором одного baseline model;
- рассмотреть SpeechBrain ECAPA-TDNN как предварительного кандидата;
- pin model revision и dependency versions;
- изолировать model loading от application services;
- добавить mocked unit tests и маленький local smoke test только на разрешенном
  audio.

Phase 7:

- собрать labeled target/non-target pair dataset;
- посчитать ROC/DET/AUC/EER/FAR/FRR;
- выбрать threshold только на dev data;
- один раз оценить held-out test data;
- только после этого обсуждать calibrated probability-like output.

## 12. Проверенные источники

- SpeechBrain ECAPA-TDNN model card:
  https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- SpeechBrain pretrained-model tutorial:
  https://speechbrain.readthedocs.io/en/stable/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.html
- SpeechBrain repository license notes:
  https://github.com/speechbrain/speechbrain
- pyannote embedding model card:
  https://huggingface.co/pyannote/embedding
- NVIDIA NeMo speaker recognition docs:
  https://docs.nvidia.com/nemo/speech/nightly/asr/speaker_recognition/intro.html
- NVIDIA NeMo speaker recognition checkpoints:
  https://docs.nvidia.com/nemo/speech/nightly/asr/speaker_recognition/results.html
- NVIDIA TitaNet-L model card:
  https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/titanet_large
- NVIDIA NeMo repository:
  https://github.com/NVIDIA-NeMo/NeMo
- NVIDIA governing terms note:
  https://docs.nvidia.com/nemo-platform/documentation/reference/eula
- WeSpeaker repository:
  https://github.com/wenet-e2e/wespeaker
