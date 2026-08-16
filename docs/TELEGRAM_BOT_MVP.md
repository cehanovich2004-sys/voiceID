# Telegram Voice Collection MVP

Status: local MVP for a first voluntary VoiceID feasibility test.

This bot collects Telegram voice messages for research data preparation only.
It is not production authentication and never returns `MATCH`, `NO_MATCH`,
probability, confidence, or an identity verdict.

## Scope

The local flow is:

```text
Telegram voice message
-> local OGG/Opus download
-> ffmpeg conversion to mono PCM16 WAV at 16000 Hz
-> pseudonymous local manifest
-> existing VoiceID feasibility probe
```

The bot does not run enrollment, anti-spoofing, model downloads, dataset
ingestion, storage of embeddings, API endpoints, UI, database services, or
production threshold selection.

## Consent Flow

`/start` explains the research purpose, Telegram transfer boundary, and
non-production status. The participant must choose `Согласен` before any audio
is accepted. Without consent, voice messages are ignored and the participant is
asked to run `/start`.

After consent, the bot creates a random pseudonymous `subject_id` and requests
six Telegram voice messages. Progress is shown as `1/6` through `6/6`.

The phrases are:

1. `Сегодня хорошая погода, и я проверяю запись своего голоса.`
2. `Сегодня хорошая погода, и я проверяю запись своего голоса.`
3. `Сегодня хорошая погода, и я проверяю запись своего голоса.`
4. `Мой голос может звучать по-разному в разные дни.`
5. `Система сравнивает особенности речи, а не содержание фразы.`
6. `Эта запись используется только для исследовательского теста.`

Only Telegram `voice` messages are accepted. Unsupported messages, unreasonable
duration, excessive file size, download failures, conversion failures, and
timeouts return generic public messages and allow the participant to repeat the
current phrase.

## Local Storage

Default data root:

```text
~/.local/share/voiceid/telegram_bot
```

The data root contains local SQLite state and audio files. It must stay outside
Git. Telegram user IDs are stored only in the SQLite state table and are not
written to the feasibility manifest. Manifest records contain pseudonymous
`sample_id`, `subject_id`, `source_group_id`, `partition`, and local `wav_path`
for the internal feasibility adapter.

The first collected session is assigned to `CALIBRATION`. The MVP does not
automatically create a `HOLDOUT` partition.

`.env`, SQLite databases, OGG/Opus files, WAV files, M4A files, and local
manifests are ignored by Git.

## Commands

Participant commands:

- `/start`: show purpose and consent buttons.
- `/status`: show progress only, without IDs or paths.
- `/restart`: delete an unfinished local session and start over.
- `/delete_me`: delete local records, audio files, manifest rows, and the
  Telegram-ID association for that user.

Operator commands:

```bash
export TELEGRAM_BOT_TOKEN=""
voiceid-telegram-bot run --data-dir ~/.local/share/voiceid/telegram_bot
```

Export a local manifest:

```bash
voiceid-telegram-bot export-manifest \
  ~/.local/share/voiceid/telegram_bot/manifests/first-test.manifest.json \
  --data-dir ~/.local/share/voiceid/telegram_bot \
  --repository-commit-sha "$(git rev-parse HEAD)"
```

Run the existing feasibility probe separately with a prepared local model cache:

```bash
python -m voiceid.calibration.feasibility \
  ~/.local/share/voiceid/telegram_bot/manifests/first-test.manifest.json \
  ~/.local/share/voiceid/telegram_bot/reports/first-test \
  --cache-dir /path/to/local/speechbrain/cache
```

The feasibility command is an operator action. The Telegram bot itself does not
load the model, compute embeddings, select a threshold, or compare identities.

## Security Notes

- `TELEGRAM_BOT_TOKEN` is read only from the environment.
- Public errors are generic and do not include token values, paths, Telegram
  payloads, pseudonymous IDs, or exception text.
- `ffmpeg` is invoked with an argument list and no shell interpolation.
- Temporary or partially converted files are removed on conversion failure.
- Audio, manifests, reports, and embeddings must not be committed.
- The bot uses Telegram polling for the MVP and does not configure webhooks.
- No analytics, telemetry, cloud storage, or third-party APIs are added beyond
  Telegram Bot API calls required for polling and file download.
