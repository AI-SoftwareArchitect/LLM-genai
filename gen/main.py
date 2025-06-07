import json
import random
from faker import Faker
from datetime import datetime, timedelta
import re

fake = Faker('tr_TR')

TEMPLATES = {
    "greetings": [
        ("merhaba", ["merhaba hoş geldin", "selam nasılsın", "merhaba sana nasıl yardımcı olabilirim"]),
        ("selam", ["selam hoş geldin", "selam nasılsın", "selam sana nasıl yardımcı olabilirim"]),
        ("günaydın", ["günaydın sana da güzel bir gün dilerim", "günaydın umarım günün güzel geçer"]),
        ("iyi akşamlar", ["iyi akşamlar sana da", "iyi akşamlar umarım günün güzel geçer"]),
        ("iyi geceler", ["iyi geceler tatlı rüyalar", "iyi geceler dinlenmeyi unutma"]),
        ("tekrar merhaba", ["tekrar hoş geldin", "yeniden merhaba"])
    ],
    "smalltalk": [
        ("nasılsın", ["teşekkür ederim iyiyim sen nasılsın", "iyiyim sen nasılsın", "çok iyiyim ya sen"]),
        ("iyi misin", ["evet iyiyim teşekkürler", "gayet iyiyim", "harikayım"]),
        ("ne yapıyorsun", ["şu anda seninle sohbet ediyorum", "seninle konuşuyorum", "burada senin için varım"]),
        ("ne düşünüyorsun", ["şu anda seninle konuşmayı düşünüyorum", "seninle sohbet etmek güzel"]),
        ("seni kim yaptı", ["beni bir yazılım geliştirici yaptı", "yapay zeka mühendisleri tarafından geliştirildim"]),
        ("kaç yaşındasın", ["benim yaşım yok ama hep buradayım", "yaşım yok ama hep öğreniyorum"])
    ],
    "info": [
        ("saat kaç", ["şu an saat {}", "şu anda saat {}", "bakıyorum saat {}"]),
        ("bugünün tarihi nedir", ["bugün {}", "tarih bugün {}", "{}"]),
        ("hangi gündeyiz", ["bugün {}", "{}"]),
        ("hangi aydayız", ["şu an {}", "{}"]),
        ("hangi yıldayız", ["şu an {}", "{}"])
    ],
    "emotion": [
        ("mutlu musun", ["benim duygularım yok ama mutlu olabilirim", "yapay zekalar mutlu olmaz ama yardımcı olabilirim"]),
        ("üzgün müsün", ["üzgün değilim ama senin için buradayım", "üzülmem ama sana destek olabilirim"]),
        ("seni seviyorum", ["ben de seni seviyorum", "teşekkür ederim ben de seni seviyorum"])
    ],
    "suggestion": [
        ("bugün ne yapsam", ["belki {} yapabilirsin", "{} öneririm", "{} iyi gelebilir"]),
        ("canım sıkılıyor", ["{} yapmayı deneyebilirsin", "biraz {} iyi gelebilir"]),
        ("bana öneri ver", ["{} öneririm", "{} yapabilirsin"]),
        ("bana bir tavsiye ver", ["{} tavsiye ederim", "{} iyi olabilir"])
    ],
    "weather": [
        ("hava nasıl", ["şu anda hava {}", "hava şu an {}", "dışarısı şu anda {} gibi görünüyor"]),
        ("bugünkü hava durumu nedir", ["hava şu an {}", "bugün hava {}", "dışarısı {}"])
    ],
    "location": [
        ("hangi şehirde yaşıyorsun", ["şu anda {}de yaşıyorum", "{}de yaşıyorum"]),
        ("nerelisin", ["{}liyim", "{}denim"]),
        ("hangi şehirde doğdun", ["{}de doğdum", "{}de dünyaya geldim"]),
        ("nerede oturuyorsun", ["{}de oturuyorum", "{}de yaşıyorum"]),
        ("hangi mahalledesin", ["{} mahallesindeyim", "{} mahallesinde oturuyorum"])
    ],
    "thanks": [
        ("teşekkürler", ["rica ederim her zaman yardımcı olurum", "bir şey değil", "her zaman"]),
        ("sağ ol", ["bir şey değil", "rica ederim"]),
        ("çok teşekkür ederim", ["ben teşekkür ederim", "yardımcı olabildiysem ne mutlu"])
    ],
    "bye": [
        ("görüşürüz", ["görüşmek üzere", "hoşça kal", "kendine iyi bak"]),
        ("hoşça kal", ["görüşmek üzere", "kendine iyi bak"]),
        ("bay bay", ["görüşmek üzere", "hoşça kal"])
    ]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_time_response():
    now = datetime.now() - timedelta(minutes=random.randint(0, 59))
    return now.strftime("%H%M")

def generate_date_response():
    today = datetime.now() - timedelta(days=random.randint(0, 365))
    return today.strftime("%d %B %Y %A").lower()

def generate_weather():
    conditions = [
        "güneşli ve sıcak", "hafif yağmurlu", "bulutlu ve serin", "rüzgarlı", "karlı", "sisli", "açık ve ılık",
        "parçalı bulutlu", "yağmurlu", "soğuk ve rüzgarlı", "ılık ve güneşli"
    ]
    return random.choice(conditions)

def generate_activity():
    activities = [
        "yeni bir kitap okumak", "uzun bir yürüyüşe çıkmak", "güzel bir film izlemek",
        "farklı bir yemek tarifi denemek", "sevdiğin müzikleri dinlemek", "biraz spor yapmak",
        "arkadaşlarınla buluşmak", "hobilerinle ilgilenmek", "resim yapmak", "yeni bir dil öğrenmek",
        "yeni bir hobi edinmek", "doğada vakit geçirmek", "yeni bir şeyler öğrenmek"
    ]
    return random.choice(activities)

def generate_city():
    return clean_text(fake.city())

def fill_template(input_text, output_text):
    şehir = generate_city()
    input_text = input_text.replace("{şehir}", şehir)
    if "{}" in output_text:
        if "saat" in input_text or "zaman" in input_text:
            output_text = output_text.format(generate_time_response())
        elif "gün" in input_text or "tarih" in input_text or "ay" in input_text or "yıl" in input_text:
            output_text = output_text.format(generate_date_response())
        elif "hava" in input_text:
            output_text = output_text.format(generate_weather())
        elif "öner" in input_text or "canım sıkılıyor" in input_text or "ne yapsam" in input_text or "tavsiye" in input_text:
            output_text = output_text.format(generate_activity())
        elif "{}de" in output_text or "{}liyim" in output_text or "{} mahallesindeyim" in output_text:
            output_text = output_text.format(şehir)
    return clean_text(input_text), clean_text(output_text)

def generate_variations(text):
    # Soru tipi, kelime sırası, eş anlamlılar, kısaltmalar
    variations = [
        ("merhaba", "selam"), ("iyi günler", "güzel günler"),
        ("bana", "lütfen bana"), ("acaba", ""), ("söyleyebilir misin", "bilgi verebilir misin"),
        ("yardımcı olur musun", "yardımcı olabilir misin"), ("tekrar", "yeniden"),
        ("bugün", "bu gün"), ("şu anda", "şimdi"), ("düşünüyorum", "merak ediyorum"),
        ("hava", "hava durumu"), ("saat", "zaman"), ("ne yapıyorsun", "ne işle meşgulsün"),
        ("nasılsın", "iyi misin"), ("teşekkürler", "sağ ol"), ("rica ederim", "bir şey değil"),
        ("sohbet", "konuşma"), ("tavsiye", "öneri"), ("akşamlar", "geceler"),
        ("neden", "niçin"), ("ne zaman", "hangi vakit"), ("kim", "kimsin")
    ]
    results = set([text])
    for old, new in variations:
        if old in text:
            results.add(text.replace(old, new, 1))
    return list(results)

def generate_dataset(num_samples=20000):
    all_pairs = set()
    for category in TEMPLATES.values():
        for input_text, output_texts in category:
            input_variants = [input_text] + generate_variations(input_text)
            for inp in input_variants:
                for out_text in output_texts:
                    for _ in range(3):  # Her varyasyon için 3 farklı paraphrase
                        inp_filled, out_filled = fill_template(inp, out_text)
                        all_pairs.add((inp_filled, out_filled))
    all_pairs = list(all_pairs)
    random.shuffle(all_pairs)

    dataset = []
    used = set()
    for inp, out in all_pairs:
        if len(dataset) >= num_samples:
            break
        key = (inp, out)
        if key not in used:
            used.add(key)
            dataset.append({"input": inp, "output": out})

    # Eksikse rastgele üret
    while len(dataset) < num_samples:
        cat = random.choice(list(TEMPLATES.keys()))
        tpl = random.choice(TEMPLATES[cat])
        inp, out_texts = tpl
        out = random.choice(out_texts)
        inp_filled, out_filled = fill_template(inp, out)
        key = (inp_filled, out_filled)
        if key not in used:
            used.add(key)
            dataset.append({"input": inp_filled, "output": out_filled})
    return dataset

def save_to_json(dataset, filename="train_data.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"{len(dataset)} adet diyalog çifti '{filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    num_samples = 20000
    dataset = generate_dataset(num_samples)
    save_to_json(dataset)

    print("\nOluşturulan örnek diyaloglar:")
    for i in range(5):
        print(f"{i+1}. Girdi: {dataset[i]['input']}")
        print(f"   Çıktı: {dataset[i]['output']}\n")