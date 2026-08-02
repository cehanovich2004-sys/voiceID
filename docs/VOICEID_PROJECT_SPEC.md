# VoiceID Project Specification

Version: 0.1 MVP

> WARNING: This file is the original product draft and is kept as historical
> context. Current technical decisions are defined by ADR files, README, and
> the current roadmap. ADR-001 supersedes the draft `app/` project layout with
> the current `src`-layout and sets the Phase 1 runtime to Python 3.11.
>
> Similarity score is not probability. Cosine similarity, confidence scores, or
> any future score must not be presented as a match probability unless it has
> been calibrated on a labeled same-speaker and different-speaker dataset.

---

# 1. Описание проекта

VoiceID — это система голосовой идентификации клиентов.

Основная идея:

Система получает голос человека во время разговора, создает голосовой embedding и сравнивает его с заранее сохраненным голосовым слепком.

Главная задача MVP:

Определить:

"Принадлежат ли две аудиозаписи одному человеку?"

---

# 2. Видение продукта

В будущем VoiceID может стать платформой для:

- банков;
- контакт-центров;
- служб поддержки;
- систем защиты от мошенничества.

Возможные будущие функции:

- проверка клиента во время звонка;
- интеграция с телефонией;
- автоматическое определение клиента;
- анализ разговоров;
- помощь оператору;
- контроль качества обслуживания.

Однако MVP НЕ должен реализовывать эти функции.

---

# 3. Команда проекта

## CEO / Founder

Отвечает за:

- бизнес;
- продукт;
- рынок;
- клиентов;
- стратегию.

---

## CTO / Architect

Отвечает за:

- архитектуру;
- технические решения;
- масштабируемость;
- контроль качества разработки.

---

## Lead Software Engineer (Codex)

Отвечает за:

- написание кода;
- реализацию задач;
- тестирование;
- документацию;
- технические предложения.

---

# 4. Главный принцип разработки

Сначала создаем простое доказательство технологии.

После подтверждения работоспособности постепенно усложняем систему.

Не создавать сложную архитектуру без необходимости.

---

# 5. MVP Scope

Версия 0.1 должна уметь:

1. Загружать аудиофайл.

2. Анализировать аудио.

3. Получать speaker embedding.

4. Сравнивать два embedding.

5. Рассчитывать similarity/confidence score.

6. Выводить результат пользователю.


Пример результата:

--------------------------------

Voice Similarity

Match confidence score:

96.7%

Decision:

MATCH

--------------------------------

---

# 6. Что НЕ входит в MVP

Не делать:

- телефонию;
- SIP;
- CRM;
- регистрацию пользователей;
- облако;
- масштабирование;
- мобильное приложение;
- коммерческий интерфейс.

---

# 7. Технологический стек

Основной язык:

Legacy draft: Python 3.12+

Superseded for Phase 1 by ADR-001: Python 3.11.

---

ML:

- PyTorch
- SpeechBrain
- pyannote.audio

---

Audio:

- librosa
- soundfile
- numpy

---

Backend:

- FastAPI

---

MVP Interface:

- Streamlit

---

Testing:

- pytest

---

# 8. Архитектура проекта

Legacy draft, superseded by ADR-001 and README for Phase 1.

voiceid/

│

├── app/

│   ├── audio/

│   ├── models/

│   ├── embeddings/

│   ├── similarity/

│   ├── services/

│   └── config/

│

├── tests/

├── docs/

├── scripts/

├── requirements.txt

├── README.md

└── main.py


---

# 9. Этапы разработки


## Phase 1

Создание структуры проекта.

Результат:

Рабочий Python проект.


---

## Phase 2

Работа с аудио.

Реализовать:

- загрузку WAV;
- проверку файла;
- получение параметров.


---

## Phase 3

Предобработка аудио.

Реализовать:

- нормализацию;
- удаление тишины;
- подготовку данных.


---

## Phase 4

Speaker Embedding.

Подключить модель.

Получать числовое представление голоса.


---

## Phase 5

Similarity Engine.

Реализовать:

- сравнение embedding;
- Cosine Similarity;
- расчет confidence score.


---

## Phase 6

MVP Interface.

Создать простой интерфейс.

---

## Phase 7

Testing.

Проверить:

- одинаковые голоса;
- разные голоса;
- шум;
- разные записи.


---

# 10. Правила разработки

Перед крупными изменениями:

1. Объяснить решение.
2. Описать альтернативы.
3. Объяснить риски.

---

Код должен быть:

- модульным;
- читаемым;
- документированным;
- расширяемым.


---

# 11. Architecture Decision Records

Все важные решения фиксировать.

Формат:

ADR-XXX

Decision:

Почему выбрано решение.

Alternative:

Какие варианты рассматривались.

Reason:

Почему выбран этот вариант.

---

# 12. Отчеты

После каждого этапа создавать отчет:

- что сделано;
- какие файлы изменены;
- какие зависимости добавлены;
- какие проблемы возникли;
- что делать дальше.


---

# 13. Главная цель

Создать работающий MVP голосовой идентификации.

Не стремиться сразу создать коммерческую систему.

Сначала доказать технологию.
