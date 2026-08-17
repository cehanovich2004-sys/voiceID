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

After consent, the bot creates a random pseudonymous `subject_id`, assigns a
stable local participant code such as `P0001`, and requests six Telegram voice
messages. Progress is shown as `1/6` through `6/6`.

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

The Telegram download path does not trust declared `file_size`, MIME type, or
filename extension. It streams the response in bounded chunks, stops as soon as
the actual byte limit is exceeded, removes partial downloads, and accepts only
content that matches the expected Telegram voice OGG/Opus container before WAV
conversion.

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
automatically create a `HOLDOUT` partition. Participant codes are stored only in
local SQLite for participant/operator communication; they are not written to the
feasibility manifest, CSV/HTML reports, or embedding metadata.

`.env`, SQLite databases, OGG/Opus files, WAV files, M4A files, and local
manifests are ignored by Git.

Manifests created by the bot are managed artifacts and must be written only
inside the bot-controlled export directory:

```text
<data-dir>/manifests
```

The export command rejects paths outside that directory, path traversal, and
symlink escapes. During `/delete_me`, each managed manifest is rewritten
atomically to remove that participant before the Telegram-ID recovery linkage is
deleted. If any managed manifest cannot be rewritten, deletion returns a generic
failure and can be safely retried.

Manual manifest copies outside `<data-dir>/manifests` are not controlled by the
bot. Operators must treat such copies as separate sensitive artifacts and delete
or regenerate them according to the data-handling procedure; the bot does not
scan the filesystem or modify unknown files.

## Commands

Participant commands:

- `/start`: show purpose and consent buttons.
- `/status`: show progress and only the caller's own `Pxxxx` code, without
  internal IDs or paths.
- `/my_code`: show only the caller's own participant code.
- `/whoami`: show only the caller's numeric Telegram user ID so the local
  operator can configure the allowlist.
- `/restart`: delete an unfinished local session and start over.
- `/delete_me`: delete local records, audio files, manifest rows, and the
  Telegram-ID association for that user.

Deletion is recoverable. If local file removal fails, the bot returns a generic
failure and keeps the Telegram-ID mapping and manifest-source rows so the
operator or participant can retry `/delete_me`. It only removes state after
local files and manifest-source rows can be deleted consistently.

Duplicate Telegram voice messages are tracked by message ID. Replayed updates
after a process restart are no-ops, and database uniqueness prevents two
parallel messages from filling the same prompt.

Operator commands:

Operator-only features are disabled unless `VOICEID_TELEGRAM_OPERATOR_IDS`
contains a comma-separated allowlist of exact positive Telegram user IDs. The
allowlist is read only from the local environment and is not logged or written to
manifests/reports. Forwarded messages, usernames, and chat IDs do not authorize
operator actions.

- `/identify`: ask an authorized operator to send one voice message for
  exploratory local identification against completed participant profiles.
- `/cancel`: cancel a pending operator identification request.

Experimental identification uses enrollment prompts `1-4` only. Prompts `5-6`
are holdout samples and are never included in the in-memory profile. The query
voice message is converted to a temporary WAV, embedded, compared to eligible
profiles, and deleted after the result or error. Embeddings stay in memory only.
The fixed exploratory policy is versioned as
`telegram-identification-policy-v1`, with threshold `0.4` and minimum margin
`0.05`. It returns only `IDENTIFIED: Pxxxx`, `UNKNOWN`, `AMBIGUOUS`,
`INVALID AUDIO`, or `IDENTIFICATION UNAVAILABLE`; it does not expose raw scores,
probability, confidence, `MATCH`, `NO_MATCH`, or a production identity decision.

```bash
export TELEGRAM_BOT_TOKEN=""
export VOICEID_TELEGRAM_OPERATOR_IDS=""
voiceid-telegram-bot run \
  --data-dir ~/.local/share/voiceid/telegram_bot \
  --model-cache-dir ~/.cache/voiceid/speechbrain_ecapa
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
- Identification model loading uses the prepared pinned local SpeechBrain cache
  in strict offline mode. Network calls during identification are disabled.
- No analytics, telemetry, cloud storage, or third-party APIs are added beyond
  Telegram Bot API calls required for polling and file download.
