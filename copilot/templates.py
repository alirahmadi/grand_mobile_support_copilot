"""Offline multilingual reply templates (used without an LLM and as the LLM fallback)."""

import re

GREETING = {
    "ru": "Здравствуйте!", "kk": "Сәлеметсіз бе!", "uz": "Assalomu alaykum!",
    "en": "Hello!", "es": "¡Hola!", "pt": "Olá!",
}

# Localised name of the violation for each concept (Russian drafts use the rule text itself).
VIOLATION = {
    "insult": {"ru": "оскорбление игроков", "kk": "ойыншыларды қорлау", "uz": "o'yinchilarni haqorat qilish",
               "en": "insulting other players", "es": "insultar a otros jugadores", "pt": "insultar outros jogadores"},
    "cheat": {"ru": "использование читов и стороннего ПО", "kk": "чит пен бөгде бағдарламаларды қолдану",
              "uz": "chit va begona dasturlardan foydalanish", "en": "use of cheats or third-party software",
              "es": "uso de trampas o software de terceros", "pt": "uso de trapaças ou software de terceiros"},
    "dm": {"ru": "убийство без RP-причины (DM)", "kk": "RP-себепсіз өлтіру (DM)", "uz": "RP sababisiz o'ldirish (DM)",
           "en": "killing without a roleplay reason (DM)", "es": "matar sin motivo de rol (DM)",
           "pt": "matar sem motivo de roleplay (DM)"},
    "nonrp": {"ru": "нарушение RP-процесса (NonRP)", "kk": "RP үдерісін бұзу (NonRP)", "uz": "RP jarayonini buzish (NonRP)",
              "en": "non-roleplay behaviour (NonRP)", "es": "comportamiento fuera de rol (NonRP)",
              "pt": "comportamento fora do roleplay (NonRP)"},
    "scam": {"ru": "обман игроков", "kk": "ойыншыларды алдау", "uz": "o'yinchilarni aldash",
             "en": "scamming other players", "es": "estafar a otros jugadores", "pt": "aplicar golpes em outros jogadores"},
    "spam_ads": {"ru": "реклама и флуд", "kk": "жарнама мен флуд", "uz": "reklama va flud",
                 "en": "advertising and flooding", "es": "publicidad y flood", "pt": "propaganda e flood"},
    "admin_abuse": {"ru": "превышение полномочий", "kk": "өкілеттікті асыра пайдалану", "uz": "vakolatni suiiste'mol qilish",
                    "en": "abuse of power", "es": "abuso de poder", "pt": "abuso de poder"},
    "threat": {"ru": "угрозы", "kk": "қорқыту", "uz": "tahdid", "en": "threats", "es": "amenazas", "pt": "ameaças"},
}
GENERIC_VIOLATION = {"ru": "нарушение правил", "kk": "ережені бұзу", "uz": "qoidabuzarlik",
                     "en": "a rule violation", "es": "una infracción de las reglas", "pt": "uma violação das regras"}

T = {
    "approved": {
        "ru": "{g} Жалоба рассмотрена. Действия игрока {nick} нарушают пункт {rules} правил: {violation}. {punish}",
        "kk": "{g} Шағымыңыз қаралды. {nick} ойыншысының әрекеті ереженің {rules}-тармағын бұзады: {violation}. {punish}",
        "uz": "{g} Shikoyatingiz ko'rib chiqildi. {nick} o'yinchisining harakatlari qoidalarning {rules}-bandini buzadi: {violation}. {punish}",
        "en": "{g} Your complaint has been reviewed. The actions of {nick} violate rule {rules}: {violation}. {punish}",
        "es": "{g} Su queja ha sido revisada. Las acciones de {nick} infringen la regla {rules}: {violation}. {punish}",
        "pt": "{g} Sua denúncia foi analisada. As ações de {nick} violam a regra {rules}: {violation}. {punish}",
    },
    "punish_known": {
        "ru": "Игрок {nick} получит наказание: {punishment}.",
        "kk": "{nick} ойыншысына жаза қолданылады: {punishment}.",
        "uz": "{nick} o'yinchisiga jazo qo'llaniladi: {punishment}.",
        "en": "{nick} will receive a penalty: {punishment}.",
        "es": "{nick} recibirá una sanción: {punishment}.",
        "pt": "{nick} receberá uma punição: {punishment}.",
    },
    "punish_unknown": {
        "ru": "Меру наказания определит администратор.",
        "kk": "Жаза түрін әкімші анықтайды.",
        "uz": "Jazo turini administrator belgilaydi.",
        "en": "The administrator will decide on the penalty.",
        "es": "El administrador decidirá la sanción.",
        "pt": "O administrador decidirá a punição.",
    },
    "rejected": {
        "ru": "{g} Жалоба рассмотрена. По предоставленным материалам нарушение правил со стороны игрока {nick} не подтверждается, жалоба отклонена.",
        "kk": "{g} Шағымыңыз қаралды. Ұсынылған материалдар бойынша {nick} ойыншысының ережені бұзғаны расталмады, шағым қабылданбады.",
        "uz": "{g} Shikoyatingiz ko'rib chiqildi. Taqdim etilgan materiallar bo'yicha {nick} o'yinchisi tomonidan qoidabuzarlik tasdiqlanmadi, shikoyat rad etildi.",
        "en": "{g} Your complaint has been reviewed. The submitted materials do not confirm a rule violation by {nick}, so the complaint is rejected.",
        "es": "{g} Su queja ha sido revisada. Los materiales enviados no confirman una infracción de {nick}, por lo que la queja se rechaza.",
        "pt": "{g} Sua denúncia foi analisada. Os materiais enviados não confirmam uma violação de {nick}, por isso a denúncia foi rejeitada.",
    },
    "need_evidence": {
        "ru": "{g} Для рассмотрения жалобы нужны доказательства нарушения: скриншоты или видеозапись, на которых видны ник нарушителя и время. Пожалуйста, подайте жалобу повторно и приложите доказательства.",
        "kk": "{g} Шағымды қарау үшін бұзушылықтың дәлелі қажет: бұзушының никі мен уақыты көрінетін скриншот немесе бейнежазба. Шағымды дәлелдерімен қайта жіберуіңізді сұраймыз.",
        "uz": "{g} Shikoyatni ko'rib chiqish uchun qoidabuzarlik dalili kerak: qoidabuzarning niki va vaqti ko'rinadigan skrinshot yoki video. Iltimos, shikoyatni dalillar bilan qayta yuboring.",
        "en": "{g} To review the complaint we need evidence of the violation: screenshots or a video showing the offender's nickname and the time. Please resubmit the complaint with evidence attached.",
        "es": "{g} Para revisar la queja necesitamos pruebas de la infracción: capturas de pantalla o un vídeo donde se vean el nick del infractor y la hora. Por favor, envíe la queja de nuevo con las pruebas.",
        "pt": "{g} Para analisar a denúncia precisamos de provas da infração: capturas de tela ou um vídeo mostrando o nick do infrator e o horário. Por favor, envie a denúncia novamente com as provas.",
    },
    "invalid_format": {
        "ru": "{g} Жалоба оформлена не по форме. Укажите, пожалуйста, точный ник нарушителя и нарушенный пункт правил, а также приложите доказательства (скриншоты или видео).",
        "kk": "{g} Шағым үлгіге сай толтырылмаған. Бұзушының нақты никін және бұзылған ереже тармағын көрсетіп, дәлелдерді (скриншот немесе бейне) тіркеңіз.",
        "uz": "{g} Shikoyat belgilangan shaklda to'ldirilmagan. Iltimos, qoidabuzarning aniq nikini va buzilgan qoida bandini ko'rsating hamda dalillarni (skrinshot yoki video) ilova qiling.",
        "en": "{g} The complaint is not filled in correctly. Please specify the offender's exact nickname and the rule that was broken, and attach evidence (screenshots or video).",
        "es": "{g} La queja no está rellenada correctamente. Indique el nick exacto del infractor y la regla infringida, y adjunte pruebas (capturas o vídeo).",
        "pt": "{g} A denúncia não foi preenchida corretamente. Informe o nick exato do infrator e a regra violada, e anexe provas (capturas de tela ou vídeo).",
    },
    "duplicate": {
        "ru": "{g} Ваше обращение по этому вопросу уже получено и находится на рассмотрении. Пожалуйста, не отправляйте повторные обращения — это не ускоряет проверку.",
        "kk": "{g} Бұл мәселе бойынша өтінішіңіз қабылданып, қаралып жатыр. Қайталама өтініш жібермеуіңізді сұраймыз — бұл тексеруді жеделдетпейді.",
        "uz": "{g} Bu masala bo'yicha murojaatingiz qabul qilingan va ko'rib chiqilmoqda. Iltimos, takroriy murojaat yubormang — bu tekshiruvni tezlashtirmaydi.",
        "en": "{g} We have already received your request about this issue and it is being reviewed. Please do not send repeated requests — it does not speed up the review.",
        "es": "{g} Ya hemos recibido su solicitud sobre este problema y está en revisión. Por favor, no envíe solicitudes repetidas: no acelera la revisión.",
        "pt": "{g} Já recebemos sua solicitação sobre este problema e ela está em análise. Por favor, não envie solicitações repetidas — isso não acelera a análise.",
    },
    "report_violation": {
        "ru": "{g} Администратор проверит ситуацию. Такие действия запрещены пунктом {rules} правил ({violation}). Если у вас есть скриншоты или видео, вы также можете подать жалобу на игрока через лаунчер.",
        "kk": "{g} Әкімші жағдайды тексереді. Мұндай әрекеттерге ереженің {rules}-тармағымен тыйым салынған ({violation}). Скриншот немесе бейнежазба болса, ойыншыға лаунчер арқылы шағым да бере аласыз.",
        "uz": "{g} Administrator vaziyatni tekshiradi. Bunday harakatlar qoidalarning {rules}-bandi bilan taqiqlangan ({violation}). Skrinshot yoki videongiz bo'lsa, o'yinchi ustidan launcher orqali shikoyat ham yuborishingiz mumkin.",
        "en": "{g} An administrator will look into the situation. Such actions are prohibited by rule {rules} ({violation}). If you have screenshots or video, you can also file a complaint against the player via the launcher.",
        "es": "{g} Un administrador revisará la situación. Estas acciones están prohibidas por la regla {rules} ({violation}). Si tiene capturas o vídeo, también puede presentar una queja contra el jugador desde el launcher.",
        "pt": "{g} Um administrador vai verificar a situação. Essas ações são proibidas pela regra {rules} ({violation}). Se tiver capturas ou vídeo, você também pode denunciar o jogador pelo launcher.",
    },
    "stuck": {
        "ru": "{g} Обращение передано администратору, он поможет вам. Если проблема сохранится, попробуйте перезайти в игру.",
        "kk": "{g} Өтінішіңіз әкімшіге жіберілді, ол сізге көмектеседі. Мәселе шешілмесе, ойынға қайта кіріп көріңіз.",
        "uz": "{g} Murojaatingiz administratorga yuborildi, u sizga yordam beradi. Muammo davom etsa, o'yinga qayta kirib ko'ring.",
        "en": "{g} Your request has been passed to an administrator who will help you. If the problem persists, try re-entering the game.",
        "es": "{g} Su solicitud se ha enviado a un administrador que le ayudará. Si el problema continúa, intente volver a entrar al juego.",
        "pt": "{g} Sua solicitação foi enviada a um administrador, que vai ajudar você. Se o problema continuar, tente entrar no jogo novamente.",
    },
    "bug": {
        "ru": "{g} Спасибо за сообщение. Опишите, пожалуйста, подробнее, что произошло, когда и на каком сервере, и при возможности приложите скриншот или видео — информация будет передана техническим специалистам.",
        "kk": "{g} Хабарламаңызға рахмет. Не болғанын, қашан және қай серверде болғанын толығырақ жазып, мүмкіндігінше скриншот немесе бейне тіркеңіз — ақпарат техникалық мамандарға жіберіледі.",
        "uz": "{g} Xabaringiz uchun rahmat. Iltimos, nima bo'lganini, qachon va qaysi serverda sodir bo'lganini batafsil yozing, imkon bo'lsa skrinshot yoki video ilova qiling — ma'lumot texnik mutaxassislarga yuboriladi.",
        "en": "{g} Thank you for the report. Please describe in more detail what happened, when and on which server, and attach a screenshot or video if possible — the information will be passed to the technical team.",
        "es": "{g} Gracias por el aviso. Describa con más detalle qué pasó, cuándo y en qué servidor, y adjunte una captura o vídeo si es posible: la información se enviará al equipo técnico.",
        "pt": "{g} Obrigado pelo aviso. Descreva com mais detalhes o que aconteceu, quando e em qual servidor, e anexe uma captura ou vídeo se possível — a informação será enviada à equipe técnica.",
    },
    "lost_items": {
        "ru": "{g} Обращение передано администрации для проверки. Укажите, пожалуйста, что именно пропало, примерное время и обстоятельства. Решение будет принято после проверки логов.",
        "kk": "{g} Өтінішіңіз тексеру үшін әкімшілікке жіберілді. Нақты не жоғалғанын, шамамен қашан және қандай жағдайда болғанын көрсетіңіз. Шешім логтар тексерілгеннен кейін қабылданады.",
        "uz": "{g} Murojaatingiz tekshirish uchun ma'muriyatga yuborildi. Iltimos, aynan nima yo'qolganini, taxminiy vaqti va holatini ko'rsating. Qaror loglar tekshirilgandan so'ng qabul qilinadi.",
        "en": "{g} Your request has been passed to the administration for review. Please specify what exactly disappeared, the approximate time and the circumstances. A decision will be made after the logs are checked.",
        "es": "{g} Su solicitud se ha enviado a la administración para su revisión. Indique qué desapareció exactamente, la hora aproximada y las circunstancias. La decisión se tomará tras revisar los registros.",
        "pt": "{g} Sua solicitação foi enviada à administração para análise. Informe o que exatamente sumiu, o horário aproximado e as circunstâncias. A decisão será tomada após a verificação dos logs.",
    },
    "payment": {
        "ru": "{g} Обращение по платежу передано администрации. Укажите, пожалуйста, дату и сумму платежа, способ оплаты и ник персонажа. Никому не сообщайте пароль и данные банковской карты.",
        "kk": "{g} Төлем бойынша өтінішіңіз әкімшілікке жіберілді. Төлемнің күні мен сомасын, төлем тәсілін және кейіпкеріңіздің никін көрсетіңіз. Құпия сөзді және банк картасының деректерін ешкімге айтпаңыз.",
        "uz": "{g} To'lov bo'yicha murojaatingiz ma'muriyatga yuborildi. Iltimos, to'lov sanasi va summasini, to'lov usulini hamda personajingiz nikini ko'rsating. Parol va bank kartasi ma'lumotlarini hech kimga bermang.",
        "en": "{g} Your payment request has been passed to the administration. Please provide the payment date and amount, the payment method and your character's nickname. Never share your password or bank card details.",
        "es": "{g} Su solicitud sobre el pago se ha enviado a la administración. Indique la fecha y el importe, el método de pago y el nick de su personaje. Nunca comparta su contraseña ni los datos de su tarjeta.",
        "pt": "{g} Sua solicitação sobre o pagamento foi enviada à administração. Informe a data e o valor, o método de pagamento e o nick do personagem. Nunca compartilhe sua senha ou os dados do cartão.",
    },
    "account": {
        "ru": "{g} Обращение о доступе к аккаунту передано администрации для срочной проверки. Пока идет проверка, смените пароль от аккаунта и привязанной почты и никому не сообщайте коды подтверждения.",
        "kk": "{g} Аккаунтқа қолжетімділік туралы өтінішіңіз жедел тексеру үшін әкімшілікке жіберілді. Тексеру кезінде аккаунт пен байланыстырылған поштаның құпия сөзін өзгертіңіз және растау кодтарын ешкімге айтпаңыз.",
        "uz": "{g} Akkauntga kirish bo'yicha murojaatingiz zudlik bilan tekshirish uchun ma'muriyatga yuborildi. Tekshiruv davomida akkaunt va unga bog'langan pochta parolini o'zgartiring hamda tasdiqlash kodlarini hech kimga bermang.",
        "en": "{g} Your account access request has been passed to the administration for urgent review. Meanwhile, change the password of your account and linked email, and never share confirmation codes with anyone.",
        "es": "{g} Su solicitud sobre el acceso a la cuenta se ha enviado a la administración para una revisión urgente. Mientras tanto, cambie la contraseña de la cuenta y del correo vinculado y no comparta los códigos de confirmación.",
        "pt": "{g} Sua solicitação sobre o acesso à conta foi enviada à administração para análise urgente. Enquanto isso, troque a senha da conta e do e-mail vinculado e não compartilhe códigos de confirmação.",
    },
    "ban_appeal": {
        "ru": "{g} Обжалования наказаний рассматривает администрация вручную. Укажите, пожалуйста, ник, дату и причину наказания, а также почему вы считаете его ошибочным. Решение будет принято после проверки.",
        "kk": "{g} Жазаға шағымдануды әкімшілік қолмен қарайды. Никіңізді, жазаның күні мен себебін және оны неге қате деп санайтыныңызды көрсетіңіз. Шешім тексеруден кейін қабылданады.",
        "uz": "{g} Jazolar ustidan shikoyatlarni ma'muriyat qo'lda ko'rib chiqadi. Iltimos, nikingizni, jazo sanasi va sababini hamda nima uchun uni noto'g'ri deb hisoblashingizni ko'rsating. Qaror tekshiruvdan so'ng qabul qilinadi.",
        "en": "{g} Penalty appeals are reviewed manually by the administration. Please provide your nickname, the date and reason for the penalty, and why you believe it was a mistake. A decision will be made after review.",
        "es": "{g} Las apelaciones de sanciones las revisa la administración manualmente. Indique su nick, la fecha y el motivo de la sanción y por qué cree que es un error. La decisión se tomará tras la revisión.",
        "pt": "{g} Os recursos contra punições são analisados manualmente pela administração. Informe seu nick, a data e o motivo da punição e por que acredita que foi um erro. A decisão será tomada após a análise.",
    },
    "admin_complaint": {
        "ru": "{g} Жалоба на действия администрации передана старшей администрации для проверки. Приложите, пожалуйста, доказательства (скриншоты или видео), если они есть.",
        "kk": "{g} Әкімшілік әрекетіне шағымыңыз тексеру үшін аға әкімшілікке жіберілді. Дәлелдеріңіз (скриншот немесе бейне) болса, тіркеңіз.",
        "uz": "{g} Ma'muriyat harakatlari ustidan shikoyatingiz tekshirish uchun katta ma'muriyatga yuborildi. Dalillaringiz (skrinshot yoki video) bo'lsa, ilova qiling.",
        "en": "{g} Your complaint about the administration's actions has been passed to senior administration for review. Please attach evidence (screenshots or video) if you have any.",
        "es": "{g} Su queja sobre la actuación de la administración se ha enviado a la administración superior para su revisión. Adjunte pruebas (capturas o vídeo) si las tiene.",
        "pt": "{g} Sua reclamação sobre a administração foi enviada à administração sênior para análise. Anexe provas (capturas ou vídeo), se tiver.",
    },
    "threat": {
        "ru": "{g} Обращение передано администрации для срочной проверки. Не сообщайте никому свои личные данные. Если вам угрожает реальная опасность, обратитесь в экстренные службы.",
        "kk": "{g} Өтінішіңіз жедел тексеру үшін әкімшілікке жіберілді. Жеке деректеріңізді ешкімге бермеңіз. Сізге шынайы қауіп төнсе, шұғыл қызметтерге хабарласыңыз.",
        "uz": "{g} Murojaatingiz zudlik bilan tekshirish uchun ma'muriyatga yuborildi. Shaxsiy ma'lumotlaringizni hech kimga bermang. Agar sizga real xavf tahdid solsa, favqulodda xizmatlarga murojaat qiling.",
        "en": "{g} Your request has been passed to the administration for urgent review. Do not share your personal data with anyone. If you are in real danger, contact emergency services.",
        "es": "{g} Su solicitud se ha enviado a la administración para una revisión urgente. No comparta sus datos personales. Si está en peligro real, contacte con los servicios de emergencia.",
        "pt": "{g} Sua solicitação foi enviada à administração para análise urgente. Não compartilhe seus dados pessoais. Se estiver em perigo real, procure os serviços de emergência.",
    },
    "question": {
        "ru": "{g} Ваш вопрос передан администратору, ответ поступит в ближайшее время.",
        "kk": "{g} Сұрағыңыз әкімшіге жіберілді, жауап жақын арада беріледі.",
        "uz": "{g} Savolingiz administratorga yuborildi, javob tez orada beriladi.",
        "en": "{g} Your question has been passed to an administrator; you will get an answer shortly.",
        "es": "{g} Su pregunta se ha enviado a un administrador; recibirá una respuesta en breve.",
        "pt": "{g} Sua pergunta foi enviada a um administrador; você receberá uma resposta em breve.",
    },
    "spam": {
        "ru": "{g} Обращение не содержит описания проблемы. Если вам нужна помощь, опишите ситуацию подробнее.",
        "kk": "{g} Өтініште мәселенің сипаттамасы жоқ. Көмек қажет болса, жағдайды толығырақ жазыңыз.",
        "uz": "{g} Murojaatda muammo tavsifi yo'q. Yordam kerak bo'lsa, vaziyatni batafsilroq yozing.",
        "en": "{g} Your request does not describe a problem. If you need help, please describe the situation in more detail.",
        "es": "{g} Su solicitud no describe ningún problema. Si necesita ayuda, describa la situación con más detalle.",
        "pt": "{g} Sua solicitação não descreve um problema. Se precisar de ajuda, descreva a situação com mais detalhes.",
    },
    "gratitude": {
        "ru": "{g} Спасибо за обратную связь! Приятной игры.",
        "kk": "{g} Пікіріңізге рахмет! Ойыныңыз сәтті болсын.",
        "uz": "{g} Fikringiz uchun rahmat! Yoqimli o'yin tilaymiz.",
        "en": "{g} Thank you for your feedback! Enjoy the game.",
        "es": "{g} ¡Gracias por sus comentarios! Disfrute del juego.",
        "pt": "{g} Obrigado pelo feedback! Bom jogo.",
    },
    "generic": {
        "ru": "{g} Ваше обращение получено и передано администратору. Ответ поступит после рассмотрения.",
        "kk": "{g} Өтінішіңіз қабылданып, әкімшіге жіберілді. Жауап қаралғаннан кейін беріледі.",
        "uz": "{g} Murojaatingiz qabul qilindi va administratorga yuborildi. Javob ko'rib chiqilgandan so'ng beriladi.",
        "en": "{g} Your request has been received and passed to an administrator. You will get a reply after review.",
        "es": "{g} Su solicitud ha sido recibida y enviada a un administrador. Recibirá respuesta tras la revisión.",
        "pt": "{g} Sua solicitação foi recebida e enviada a um administrador. Você receberá uma resposta após a análise.",
    },
}

# ---------- punishment translation (Russian strings mined from admin replies) ----------
_P_TYPES = [
    (r"мут|mute|затычк", {"ru": "мут", "kk": "мут (чатқа тыйым)", "uz": "mut (chat taqiqi)", "en": "mute", "es": "silencio (mute)", "pt": "mute"}),
    (r"деморган|джайл|jail|тюрьм|демор", {"ru": "деморган", "kk": "деморган (түрме)", "uz": "demorgan (qamoq)", "en": "jail (Demorgan)", "es": "cárcel (Demorgan)", "pt": "prisão (Demorgan)"}),
    (r"варн|warn|предупрежд", {"ru": "предупреждение", "kk": "ескерту (варн)", "uz": "ogohlantirish (varn)", "en": "warning", "es": "advertencia", "pt": "advertência"}),
    (r"бан|ban|блокир", {"ru": "блокировка", "kk": "бұғаттау (бан)", "uz": "bloklash (ban)", "en": "ban", "es": "bloqueo (ban)", "pt": "banimento"}),
    (r"кик|kick", {"ru": "кик", "kk": "серверден шығару (кик)", "uz": "serverdan chiqarish (kik)", "en": "kick", "es": "expulsión (kick)", "pt": "expulsão (kick)"}),
]
_P_UNITS = [
    (r"мин", {"kk": "{n} минутқа", "uz": "{n} daqiqaga", "en": "for {n} minutes", "es": "por {n} minutos", "pt": "por {n} minutos"}),
    (r"час", {"kk": "{n} сағатқа", "uz": "{n} soatga", "en": "for {n} hours", "es": "por {n} horas", "pt": "por {n} horas"}),
    (r"дн|ден|сут", {"kk": "{n} күнге", "uz": "{n} kunga", "en": "for {n} days", "es": "por {n} días", "pt": "por {n} dias"}),
]
_PERMA = {"kk": "мерзімсіз", "uz": "muddatsiz", "en": "permanently", "es": "permanente", "pt": "permanente"}


def translate_punishment(p: str, lang: str) -> str | None:
    """'мут на 30 минут' -> 'мут (чатқа тыйым) 30 минутқа'. None if it cannot be parsed safely."""
    if not p:
        return None
    if lang == "ru":
        return p.strip().rstrip(".")
    low = p.lower()
    parts = []
    for chunk in re.split(r"[,;+]| и ", low):
        typ = next((names for pat, names in _P_TYPES if re.search(pat, chunk)), None)
        if not typ:
            return None
        name = typ.get(lang, typ["en"])
        m = re.search(r"(\d+)\s*([а-яa-z]+)", chunk)
        if m:
            unit = next((u for pat, u in _P_UNITS if re.match(pat, m.group(2))), None)
            if not unit:
                return None
            dur = unit.get(lang, unit["en"]).format(n=m.group(1))
            parts.append(f"{dur} {name}" if lang in ("uz",) else f"{name} {dur}")
        elif re.search(r"навсегда|перм|perm|forever", chunk):
            parts.append(f"{_PERMA.get(lang, 'permanently')} {name}" if lang == "uz" else f"{name} {_PERMA.get(lang, 'permanently')}")
        else:
            parts.append(name)
    return ", ".join(parts) if parts else None


# ---------- Uzbek Latin -> Cyrillic (reply in the script the player used) ----------
_UZ_MULTI = [("o'", "ў"), ("O'", "Ў"), ("g'", "ғ"), ("G'", "Ғ"), ("sh", "ш"), ("Sh", "Ш"), ("SH", "Ш"),
             ("ch", "ч"), ("Ch", "Ч"), ("CH", "Ч"), ("yo", "ё"), ("Yo", "Ё"), ("yu", "ю"), ("Yu", "Ю"),
             ("ya", "я"), ("Ya", "Я"), ("ye", "е"), ("Ye", "Е")]
_UZ_SINGLE = dict(zip("abdefghijklmnopqrstuvxyzABDEFGHIJKLMNOPQRSTUVXYZ",
                      "абдефгҳижклмнопқрстувхйзАБДЕФГҲИЖКЛМНОПҚРСТУВХЙЗ"))


def uz_latin_to_cyrillic(text: str) -> str:
    def conv(word: str) -> str:
        if "_" in word or any(ch.isdigit() for ch in word):
            return word  # nicknames, rule ids
        w = re.sub(r"^e", "э", word)
        w = re.sub(r"^E", "Э", w)
        for a, b in _UZ_MULTI:
            w = w.replace(a, b)
        w = w.replace("'", "ъ")
        return "".join(_UZ_SINGLE.get(ch, ch) for ch in w)
    return re.sub(r"[A-Za-z'_0-9]+", lambda m: conv(m.group(0)), text)


def render(key: str, lang: str, **kw) -> str:
    lang = lang if lang in GREETING else "ru"
    tpl = T.get(key, T["generic"])
    s = tpl.get(lang) or tpl.get("en") or tpl["ru"]
    kw.setdefault("g", GREETING[lang])
    out = s.format(**kw)
    return re.sub(r"\s+", " ", out).strip()
