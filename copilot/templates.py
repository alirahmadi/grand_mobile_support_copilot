"""Offline reply templates in the style of real Grand Mobile admin replies (ru/kk/uz/en, es/pt -> en).

Complaints are answered like the launcher templates («Уважаемый игрок, благодарим вас за жалобу…»),
in-game reports like the short in-game replies («Приветствую, слежу за вами.»).
"""

import re

GREETING = {"ru": "Приветствую!", "kk": "Сәлеметсіз бе!", "uz": "Assalomu alaykum!", "en": "Hello!", "es": "¡Hola!", "pt": "Olá!"}

C = {  # complaints from the launcher
    "approved": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Жалоба рассмотрена и одобрена: игрок {nick} получит {punishment} за нарушение пункта {ref}.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Шағым қаралып, мақұлданды: {nick} ойыншысы ереженің {ref} тармағын бұзғаны үшін жаза алады — {punishment}.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. Shikoyat ko'rib chiqildi va qanoatlantirildi: {nick} qoidalarning {ref} bandini buzgani uchun jazo oladi — {punishment}.",
        "en": "Dear player, thank you for your complaint. It has been reviewed and approved: {nick} will receive {punishment} for violating rule {ref}.",
    },
    "approved_no_punishment": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Жалоба рассмотрена и одобрена: к игроку {nick} будут применены меры за нарушение пункта {ref}.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Шағым қаралып, мақұлданды: {nick} ойыншысына ереженің {ref} тармағын бұзғаны үшін шара қолданылады.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. Shikoyat ko'rib chiqildi va qanoatlantirildi: {nick} qoidalarning {ref} bandini buzgani uchun unga nisbatan choralar ko'riladi.",
        "en": "Dear player, thank you for your complaint. It has been reviewed and approved: action will be taken against {nick} for violating rule {ref}.",
    },
    "need_evidence": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Для принятия мер нам недостаточно доказательств: приложите видеозапись или необрезанные скриншоты, на которых видны ник нарушителя и сам момент нарушения (п. 1.2.2 правил подачи жалоб). Пожалуйста, подайте новую жалобу с доказательствами.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Шара қолдану үшін дәлел жеткіліксіз: бұзушының никі мен бұзушылық сәті көрінетін бейнежазбаны немесе қиылмаған скриншоттарды тіркеңіз (шағым беру ережесінің 1.2.2-тармағы). Дәлелдерімен жаңа шағым жіберуіңізді сұраймыз.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. Choralar ko'rish uchun dalillar yetarli emas: qoidabuzarning niki va qoidabuzarlik lahzasi ko'rinadigan video yoki kesilmagan skrinshotlarni ilova qiling (shikoyat berish qoidalarining 1.2.2-bandi). Iltimos, dalillar bilan yangi shikoyat yozing.",
        "en": "Dear player, thank you for your complaint. The evidence is not sufficient to take action: please attach a video or uncropped screenshots showing the offender's nickname and the moment of the violation (complaint rules, 1.2.2) and submit a new complaint.",
    },
    "invalid_profanity": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Жалоба оформлена с нарушением п. 1.2.7 правил подачи жалоб (оскорбления и нецензурная лексика в тексте) и рассмотрению не подлежит.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Шағым шағым беру ережесінің 1.2.7-тармағын бұза отырып жазылған (мәтінде балағат сөздер бар), сондықтан қаралмайды.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. Shikoyat shikoyat berish qoidalarining 1.2.7-bandini buzgan holda yozilgan (matnda haqorat va so'kinish bor), shuning uchun ko'rib chiqilmaydi.",
        "en": "Dear player, thank you for your complaint. It violates rule 1.2.7 of the complaint rules (insults and obscene language in the text) and will not be considered.",
    },
    "invalid_description": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. В жалобе нет точного описания произошедшего (п. 1.2.3 правил подачи жалоб): укажите ник нарушителя, что именно произошло и приложите доказательства, затем подайте жалобу заново.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Шағымда не болғаны нақты сипатталмаған (шағым беру ережесінің 1.2.3-тармағы): бұзушының никін, нақты не болғанын көрсетіп, дәлелдерді тіркеп, шағымды қайта жіберіңіз.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. Shikoyatda nima sodir bo'lgani aniq yozilmagan (shikoyat berish qoidalarining 1.2.3-bandi): qoidabuzarning nikini, aynan nima bo'lganini ko'rsating, dalillarni ilova qilib, shikoyatni qayta yuboring.",
        "en": "Dear player, thank you for your complaint. It lacks a clear description of what happened (complaint rules, 1.2.3): please specify the offender's nickname and what exactly happened, attach evidence and submit it again.",
    },
    "not_a_complaint": {
        "ru": "Уважаемый игрок, раздел жалоб не предназначен для вопросов и технических проблем (п. 1.1.2 правил подачи жалоб). С таким вопросом обратитесь в техническую поддержку или в Discord игры.",
        "kk": "Құрметті ойыншы, шағымдар бөлімі сұрақтар мен техникалық мәселелерге арналмаған (шағым беру ережесінің 1.1.2-тармағы). Мұндай сұрақпен техникалық қолдау қызметіне немесе ойынның Discord-серверіне жүгініңіз.",
        "uz": "Hurmatli o'yinchi, shikoyatlar bo'limi savollar va texnik muammolar uchun mo'ljallanmagan (shikoyat berish qoidalarining 1.1.2-bandi). Bunday savol bilan texnik yordamga yoki o'yinning Discord-serveriga murojaat qiling.",
        "en": "Dear player, the complaints section is not intended for questions or technical issues (complaint rules, 1.1.2). Please contact technical support or the game's Discord.",
    },
    "duplicate": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Данная жалоба является дубликатом предыдущей (п. 1.2.9 правил подачи жалоб) и закрыта. Пожалуйста, ожидайте рассмотрения первой жалобы.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Бұл шағым алдыңғысын қайталайды (шағым беру ережесінің 1.2.9-тармағы), сондықтан жабылды. Алғашқы шағымның қаралуын күтіңіз.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. Ushbu shikoyat avvalgisining takroridir (shikoyat berish qoidalarining 1.2.9-bandi) va yopildi. Iltimos, birinchi shikoyat ko'rib chiqilishini kuting.",
        "en": "Dear player, thank you for your complaint. It duplicates a previous one (complaint rules, 1.2.9) and has been closed. Please wait for the first complaint to be reviewed.",
    },
    "already_punished": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Игрок {nick} уже был наказан за данное нарушение, повторное наказание не предусмотрено.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. {nick} ойыншысы бұл бұзушылық үшін жазаланып қойған, қайта жазалау қарастырылмаған.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. {nick} bu qoidabuzarlik uchun allaqachon jazolangan, takroriy jazo nazarda tutilmagan.",
        "en": "Dear player, thank you for your complaint. {nick} has already been punished for this violation; a second punishment is not applied.",
    },
    "rejected": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Жалоба рассмотрена и отклонена: по предоставленным доказательствам нарушение правил со стороны игрока {nick} не подтверждается.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Шағым қаралып, қабылданбады: ұсынылған дәлелдер бойынша {nick} ойыншысының ережені бұзғаны расталмады.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. Shikoyat ko'rib chiqildi va rad etildi: taqdim etilgan dalillar bo'yicha {nick} tomonidan qoidabuzarlik tasdiqlanmadi.",
        "en": "Dear player, thank you for your complaint. It has been reviewed and rejected: the evidence provided does not confirm a rule violation by {nick}.",
    },
    "appeal": {
        "ru": "Уважаемый игрок, ваша жалоба на выданное наказание передана старшей администрации. Решение будет принято после проверки доказательств администратора, выдавшего наказание.",
        "kk": "Құрметті ойыншы, берілген жазаға қатысты шағымыңыз аға әкімшілікке жіберілді. Шешім жазаны берген әкімшінің дәлелдері тексерілгеннен кейін қабылданады.",
        "uz": "Hurmatli o'yinchi, berilgan jazo ustidan shikoyatingiz katta ma'muriyatga yuborildi. Qaror jazo bergan administratorning dalillari tekshirilgandan so'ng qabul qilinadi.",
        "en": "Dear player, your appeal against the penalty has been passed to senior administration. A decision will be made after checking the evidence of the administrator who issued it.",
    },
    "admin_complaint": {
        "ru": "Уважаемый игрок, ваша жалоба на действия администратора передана старшей администрации. Решение будет принято после проверки.",
        "kk": "Құрметті ойыншы, әкімшінің әрекетіне шағымыңыз аға әкімшілікке жіберілді. Шешім тексеруден кейін қабылданады.",
        "uz": "Hurmatli o'yinchi, administrator harakatlari ustidan shikoyatingiz katta ma'muriyatga yuborildi. Qaror tekshiruvdan so'ng qabul qilinadi.",
        "en": "Dear player, your complaint about the administrator has been passed to senior administration. A decision will be made after review.",
    },
    "generic_complaint": {
        "ru": "Уважаемый игрок, благодарим вас за жалобу. Она передана администрации и будет рассмотрена в течение 48 часов.",
        "kk": "Құрметті ойыншы, шағымыңыз үшін рахмет. Ол әкімшілікке жіберілді және 48 сағат ішінде қаралады.",
        "uz": "Hurmatli o'yinchi, shikoyatingiz uchun rahmat. U ma'muriyatga yuborildi va 48 soat ichida ko'rib chiqiladi.",
        "en": "Dear player, thank you for your complaint. It has been passed to the administration and will be reviewed within 48 hours.",
    },
}

R = {  # in-game reports
    "watching_you": {"ru": "Приветствую, слежу за вами.", "kk": "Сәлеметсіз бе, сізді бақылап отырмын.",
                     "uz": "Assalomu alaykum, kuzatyapman.", "en": "Hello, I'm watching you."},
    "watching_player": {"ru": "Приветствую, слежу за игроком.", "kk": "Сәлеметсіз бе, ойыншыны бақылап отырмын.",
                        "uz": "Assalomu alaykum, o'yinchini kuzatyapman.", "en": "Hello, I'm watching the player."},
    "help": {"ru": "Приветствую, спешу к вам на помощь.", "kk": "Сәлеметсіз бе, көмекке келе жатырмын.",
             "uz": "Assalomu alaykum, yordamga boryapman.", "en": "Hello, I'm on my way to help you."},
    "offtopic": {"ru": "Приветствую, уточните ваш вопрос или жалобу.", "kk": "Сәлеметсіз бе, сұрағыңызды немесе шағымыңызды нақтылаңыз.",
                 "uz": "Assalomu alaykum, savolingiz yoki shikoyatingizni aniqlashtiring.", "en": "Hello, please clarify your question or complaint."},
    "freebie": {"ru": "Приветствую, не предоставляем данную услугу: репорт не предназначен для просьб о валюте, уровнях и имуществе (п. 3.1 ОП).",
                "kk": "Сәлеметсіз бе, мұндай қызмет көрсетілмейді: репорт ақша, деңгей немесе мүлік сұрауға арналмаған (3.1 ОП тармағы).",
                "uz": "Assalomu alaykum, bunday xizmat ko'rsatilmaydi: report pul, daraja yoki mulk so'rash uchun emas (3.1 ОП bandi).",
                "en": "Hello, we don't provide this service: reports are not for requesting money, levels or property (rule 3.1 ОП)."},
    "question": {"ru": "Приветствую, ваш вопрос передал. Ожидайте!", "kk": "Сәлеметсіз бе, сұрағыңызды жеткіздім. Күтіңіз!",
                 "uz": "Assalomu alaykum, savolingizni yetkazdim. Kuting!", "en": "Hello, I've passed your question on. Please wait!"},
    "bug": {"ru": "Приветствую, опишите подробнее, что произошло и когда. Если проблема не решится, обратитесь в техническую поддержку.",
            "kk": "Сәлеметсіз бе, не болғанын және қашан болғанын толығырақ жазыңыз. Мәселе шешілмесе, техникалық қолдау қызметіне жүгініңіз.",
            "uz": "Assalomu alaykum, nima bo'lganini va qachon bo'lganini batafsil yozing. Muammo hal bo'lmasa, texnik yordamga murojaat qiling.",
            "en": "Hello, please describe what happened and when. If the problem persists, contact technical support."},
    "payment": {"ru": "Приветствую, если оплата не поступила, обратитесь в техническую поддержку и приложите чек. Никому не сообщайте данные карты и пароль.",
                "kk": "Сәлеметсіз бе, төлем түспесе, чекті тіркеп техникалық қолдау қызметіне жүгініңіз. Карта деректері мен құпия сөзді ешкімге айтпаңыз.",
                "uz": "Assalomu alaykum, to'lov tushmagan bo'lsa, chekni ilova qilib texnik yordamga murojaat qiling. Karta ma'lumotlari va parolni hech kimga bermang.",
                "en": "Hello, if the payment hasn't arrived, contact technical support and attach the receipt. Never share your card details or password."},
    "account": {"ru": "Приветствую, по вопросам доступа к аккаунту и пропажи имущества обратитесь в техническую поддержку. Смените пароль и никому не сообщайте коды подтверждения.",
                "kk": "Сәлеметсіз бе, аккаунтқа кіру және мүліктің жоғалуы бойынша техникалық қолдау қызметіне жүгініңіз. Құпия сөзді ауыстырып, растау кодтарын ешкімге айтпаңыз.",
                "uz": "Assalomu alaykum, akkauntga kirish va mulk yo'qolishi bo'yicha texnik yordamga murojaat qiling. Parolni almashtiring va tasdiqlash kodlarini hech kimga bermang.",
                "en": "Hello, for account access and missing property please contact technical support. Change your password and never share confirmation codes."},
    "appeal_report": {"ru": "Приветствую, обжаловать наказание можно через раздел жалоб в лаунчере в течение 48 часов с момента его выдачи.",
                      "kk": "Сәлеметсіз бе, жазаға лаунчердегі шағымдар бөлімі арқылы ол берілген сәттен бастап 48 сағат ішінде шағымдана аласыз.",
                      "uz": "Assalomu alaykum, jazo ustidan launcherdagi shikoyatlar bo'limi orqali u berilgan paytdan boshlab 48 soat ichida shikoyat qilishingiz mumkin.",
                      "en": "Hello, you can appeal a penalty through the complaints section of the launcher within 48 hours of it being issued."},
    "threat": {"ru": "Приветствую, передал вашу жалобу старшей администрации. Не сообщайте никому личные данные.",
               "kk": "Сәлеметсіз бе, шағымыңызды аға әкімшілікке жеткіздім. Жеке деректеріңізді ешкімге айтпаңыз.",
               "uz": "Assalomu alaykum, shikoyatingizni katta ma'muriyatga yetkazdim. Shaxsiy ma'lumotlaringizni hech kimga bermang.",
               "en": "Hello, I've passed your report to senior administration. Do not share your personal data with anyone."},
    "gratitude": {"ru": "Приветствую, приятной игры на Grand Mobile!", "kk": "Сәлеметсіз бе, Grand Mobile-да ойыныңыз сәтті болсын!",
                  "uz": "Assalomu alaykum, Grand Mobile'da yoqimli o'yin tilaymiz!", "en": "Hello, enjoy playing Grand Mobile!"},
    "generic": {"ru": "Приветствую, ожидайте.", "kk": "Сәлеметсіз бе, күтіңіз.", "uz": "Assalomu alaykum, kuting.", "en": "Hello, please wait."},
}
T = {**C, **R}

# ---------- punishment normalisation / translation (strings mined from admin replies) ----------
_P_TYPES = [
    (r"блокировк\w*\s+(игрового\s+)?чата|мут|mute|затычк", {"ru": "мут", "kk": "мут (чатқа тыйым)", "uz": "mut (chat taqiqi)", "en": "a mute"}),
    (r"деморган|джайл|jail|тюрьм|демор", {"ru": "деморган", "kk": "деморган (түрме)", "uz": "demorgan (qamoq)", "en": "jail (Demorgan)"}),
    (r"варн|warn", {"ru": "варн", "kk": "варн (ескерту)", "uz": "varn (ogohlantirish)", "en": "a warning (WARN)"}),
    (r"предупрежд", {"ru": "предупреждение", "kk": "ескерту", "uz": "ogohlantirish", "en": "a warning"}),
    (r"блокировк|бан|ban", {"ru": "бан", "kk": "бан (бұғаттау)", "uz": "ban (bloklash)", "en": "a ban"}),
    (r"кик|kick", {"ru": "кик", "kk": "серверден шығару (кик)", "uz": "serverdan chiqarish (kik)", "en": "a kick"}),
]
_RU_UNITS = [(r"мин", ("минуту", "минуты", "минут")), (r"час", ("час", "часа", "часов")), (r"(дн|ден|сут)", ("день", "дня", "дней"))]
_P_UNITS = [
    (r"мин|daqiq|minut", {"kk": "{n} минутқа", "uz": "{n} daqiqaga", "en": "for {n} minutes"}),
    (r"час|soat|hour", {"kk": "{n} сағатқа", "uz": "{n} soatga", "en": "for {n} hours"}),
    (r"дн|ден|сут|kun|day", {"kk": "{n} күнге", "uz": "{n} kunga", "en": "for {n} days"}),
]
_PERMA = {"ru": "навсегда", "kk": "мерзімсіз", "uz": "muddatsiz", "en": "permanently"}


def _ru_plural(n: int, forms: tuple) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def translate_punishment(p: str, lang: str) -> str | None:
    """'блокировкой чата на 30 минут' -> ru 'мут на 30 минут', kk 'мут (чатқа тыйым) 30 минутқа'. None if unparseable."""
    if not p:
        return None
    low = p.lower().strip()
    if lang == "uz" and re.search(r"(kunga|daqiqaga|soatga)", low):
        return p.strip()  # the uz server already writes punishments in Uzbek
    typ = next((names for pat, names in _P_TYPES if re.search(pat, low)), None)
    if not typ:
        return None if lang != "ru" else p.strip().rstrip(".")
    name = typ.get(lang, typ["en"])
    m = re.search(r"(\d+)\s*([а-яa-z']+)", low)
    if re.search(r"навсегда|перм|perm|forever", low):
        return f"{_PERMA['uz']} {name}" if lang == "uz" else f"{name} {_PERMA.get(lang, 'permanently')}"
    if not m:
        return name
    n, unit_word = int(m.group(1)), m.group(2)
    if lang == "ru":
        forms = next((f for pat, f in _RU_UNITS if re.match(pat, unit_word)), None)
        return f"{name} на {n} {_ru_plural(n, forms)}" if forms else p.strip().rstrip(".")
    unit = next((u for pat, u in _P_UNITS if re.match(pat, unit_word)), None)
    if not unit:
        return None
    dur = unit.get(lang, unit["en"]).format(n=n)
    return f"{dur} {name}" if lang == "uz" else f"{name} {dur}"


# ---------- Uzbek Latin -> Cyrillic (reply in the script the player used) ----------
_UZ_MULTI = [("o'", "ў"), ("O'", "Ў"), ("g'", "ғ"), ("G'", "Ғ"), ("sh", "ш"), ("Sh", "Ш"), ("SH", "Ш"),
             ("ch", "ч"), ("Ch", "Ч"), ("CH", "Ч"), ("yo", "ё"), ("Yo", "Ё"), ("yu", "ю"), ("Yu", "Ю"),
             ("ya", "я"), ("Ya", "Я"), ("ye", "е"), ("Ye", "Е")]
_UZ_SINGLE = dict(zip("abdefghijklmnopqrstuvxyzABDEFGHIJKLMNOPQRSTUVXYZ",
                      "абдефгҳижклмнопқрстувхйзАБДЕФГҲИЖКЛМНОПҚРСТУВХЙЗ"))


def uz_latin_to_cyrillic(text: str) -> str:
    def conv(word: str) -> str:
        if "_" in word or any(ch.isdigit() for ch in word) or word in ("Grand", "Mobile", "Discord"):
            return word  # nicknames, rule ids, names
        w = re.sub(r"^e", "э", word)
        w = re.sub(r"^E", "Э", w)
        for a, b in _UZ_MULTI:
            w = w.replace(a, b)
        w = w.replace("'", "ъ")
        return "".join(_UZ_SINGLE.get(ch, ch) for ch in w)
    return re.sub(r"[A-Za-z'_0-9]+", lambda m: conv(m.group(0)), text)


def render(key: str, lang: str, **kw) -> str:
    tpl = T.get(key, T["generic"])
    s = tpl.get(lang) or tpl.get("en") or tpl["ru"]
    kw.setdefault("nick", "")
    out = s.format(**kw)
    out = re.sub(r"\s+([,.:;])", r"\1", re.sub(r"\s+", " ", out))
    return out.strip()
