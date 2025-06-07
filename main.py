import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import argparse
from collections import deque
from datetime import datetime
import random
import re
from difflib import SequenceMatcher
import sys

# -------------------- Enhanced Dataset --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini sil
    return text

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

class ChatDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        # Küçük harf ve noktalama temizliği
        for pair in self.data:
            pair['input'] = clean_text(pair['input'])
            pair['output'] = clean_text(pair['output'])
        self.vocab = set()
        for pair in self.data:
            self.vocab.update(pair['input'].split())
            self.vocab.update(pair['output'].split())
        self.word2idx = {w: i+4 for i, w in enumerate(sorted(self.vocab))}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        self.word2idx['<SOS>'] = 2
        self.word2idx['<EOS>'] = 3
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp = [self.word2idx['<SOS>']] + [self.word2idx.get(w, 1) for w in self.data[idx]['input'].split()] + [self.word2idx['<EOS>']]
        out = [self.word2idx['<SOS>']] + [self.word2idx.get(w, 1) for w in self.data[idx]['output'].split()] + [self.word2idx['<EOS>']]
        return torch.tensor(inp), torch.tensor(out)

# -------------------- Attention Mechanism --------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.permute(1, 0, 2)  # [batch, 1, hidden_dim]
        hidden = hidden.repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, seq_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [batch, seq_len]
        return F.softmax(attention, dim=1)  # softmax over seq_len

# -------------------- Enhanced Model with Attention --------------------
class BrainAgent(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, emotion_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embed_dim + hidden_dim + emotion_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.emotion_emb = nn.Embedding(emotion_dim, emotion_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src, tgt, emotion, teacher_forcing_ratio=0.7):
        embedded_src = self.dropout(self.embed(src))
        encoder_outputs, hidden = self.encoder(embedded_src)
        batch_size = src.size(0)
        max_len = tgt.size(1)
        vocab_size = self.out.out_features
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(src.device)
        decoder_input = tgt[:, 0].unsqueeze(1)
        emo_emb = self.emotion_emb(emotion).unsqueeze(1)
        for t in range(1, max_len):
            embedded = self.dropout(self.embed(decoder_input))
            attn_weights = self.attention(hidden, encoder_outputs)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            decoder_input_cat = torch.cat((embedded, attn_applied, emo_emb), dim=2)
            output, hidden = self.decoder(decoder_input_cat, hidden)
            output = self.out(output.squeeze(1))
            outputs[t] = output
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        return outputs.permute(1, 0, 2)

# -------------------- Working Memory --------------------
class WorkingMemory:
    def __init__(self, maxlen=5):
        self.context = deque(maxlen=maxlen)
    def update(self, user_input, bot_response):
        self.context.append({'user': user_input, 'bot': bot_response, 'time': datetime.now().strftime("%H:%M:%S")})
    def get_context(self):
        # Son n diyalogu birleştirip döndür
        return " ".join([f"kullanıcı: {x['user']} bot: {x['bot']}" for x in self.context])

# -------------------- Metacognition Module --------------------
class Metacognition:
    def __init__(self):
        self.error_log = []
        self.self_reflections = []
        
    def analyze_response(self, user_input, bot_response):
        if not bot_response.strip():
            self.log_error("BOS_YANIT", user_input, bot_response)
            return False
        return True
    
    def log_error(self, error_type, user_input, bot_response):
        self.error_log.append({
            'type': error_type,
            'input': user_input,
            'output': bot_response,
            'time': datetime.now().strftime("%H:%M:%S")
        })
        
    def get_insights(self):
        if not self.error_log:
            return "Son 10 hatada sorun tespit edilmedi"
        last_errors = self.error_log[-10:]
        error_counts = {}
        for error in last_errors:
            error_counts[error['type']] = error_counts.get(error['type'], 0) + 1
        insights = []
        for err_type, count in error_counts.items():
            if count > 3:
                insights.append(f"{err_type} hatası {count} kez tekrarlandı")
        return " | ".join(insights) if insights else "Önemli hata örüntüsü yok"

# -------------------- Online Learning (Nöroplastisite) --------------------
class OnlineLearner:
    def __init__(self, model, optimizer, loss_fn, vocab_size):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.vocab_size = vocab_size
        self.adaptation_threshold = 0.7
        
    def adapt(self, user_input, corrected_output, word2idx):
        input_tensor = torch.tensor([[word2idx.get(w, 1) for w in ["<SOS>"] + user_input.split() + ["<EOS>"]]])
        target_tensor = torch.tensor([[word2idx.get(w, 1) for w in ["<SOS>"] + corrected_output.split() + ["<EOS>"]]])
        self.optimizer.zero_grad()
        output = self.model(input_tensor, target_tensor[:, :-1], torch.tensor([0]))
        loss = self.loss_fn(output.reshape(-1, self.vocab_size), target_tensor[:, 1:].reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

LEARNING_RATE = 0.0005
# -------------------- Enhanced Chat Manager --------------------
class BrainChat:
    def __init__(self, model, dataset, device, ltm_path="long_term_memory.json"):
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset
        self.word2idx = dataset.word2idx
        self.idx2word = dataset.idx2word
        self.ltm_path = ltm_path
        self.ltm = self.load_ltm()
        self.emotions = {"neutral": 0, "happy":1, "sad":2, "angry":3, "excited":4}
        self.current_emotion = self.emotions["neutral"]
        self.working_memory = WorkingMemory()
        self.metacognition = Metacognition()
        self.online_learner = OnlineLearner(
            model=self.model,
            optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
            loss_fn=nn.CrossEntropyLoss(ignore_index=0),
            vocab_size=dataset.vocab_size
        )

    def load_ltm(self):
        if os.path.exists(self.ltm_path):
            with open(self.ltm_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_ltm(self):
        with open(self.ltm_path, "w", encoding="utf-8") as f:
            json.dump(self.ltm, f, ensure_ascii=False, indent=2)

    def update_ltm(self, inp, outp):
        self.ltm[inp] = outp
        self.save_ltm()

    def set_emotion(self, emo_str):
        self.current_emotion = self.emotions.get(emo_str, 0)

    def preprocess(self, text):
        text = clean_text(text)
        return [self.word2idx.get(w, 1) for w in text.split()]

    def postprocess(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices if idx not in [0, 2, 3]]
        return " ".join(words)

    def respond(self, user_input):
        user_input = clean_text(user_input)
        context = self.working_memory.get_context()
        full_input = (context + " " + user_input).strip() if context else user_input

        # LTM'de kontrol et (benzerlik eşiği ile)
        for k in self.ltm:
            if similar(k, full_input) > 0.85:
                response = self.ltm[k]
                self.working_memory.update(user_input, response)
                return response

        # Modelden cevap üret (seq2seq decoding)
        inp = [self.word2idx['<SOS>']] + [self.word2idx.get(w, 1) for w in full_input.split()] + [self.word2idx['<EOS>']]
        inp_tensor = torch.tensor([inp], dtype=torch.long).to(self.device)
        emotion_tensor = torch.tensor([self.current_emotion], dtype=torch.long).to(self.device)
        self.model.eval()
        with torch.no_grad():
            decoder_input = torch.tensor([[self.word2idx['<SOS>']]], dtype=torch.long).to(self.device)
            outputs = []
            for _ in range(30):
                out = self.model(inp_tensor, decoder_input, emotion_tensor, teacher_forcing_ratio=0)
                next_token = out[:, -1, :].argmax(dim=-1)
                token_id = next_token.item()
                if token_id == self.word2idx['<EOS>']:
                    break
                outputs.append(token_id)
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
            response = self.postprocess(outputs)

        self.working_memory.update(user_input, response)

        # Eğer cevap boşsa kullanıcıdan doğru cevabı al ve LTM'ye ekle
        if not response.strip():
            print("Bot: Bunu bilmiyorum. Lütfen doğru cevabı gir, hemen öğreneceğim:")
            new_answer = input("Sen (doğru cevap): ").strip()
            new_answer = clean_text(new_answer)
            self.update_ltm(full_input, new_answer)
            self.working_memory.update(user_input, new_answer)
            return f"Teşekkürler, öğrendim! Bundan sonra bu soruya böyle cevap vereceğim."
        else:
            # Modelin cevabını da LTM'ye ekle (isteğe bağlı, çok tekrar olmasın diye kapalı bırakabilirsin)
            # self.update_ltm(full_input, response)
            return response

# -------------------- Training Function --------------------
def train(data_path, epochs, batch_size, lr, device):
    dataset = ChatDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = BrainAgent(dataset.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            emotion_tensor = torch.zeros(src.size(0), dtype=torch.long).to(device)  # <-- DÜZELTME
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1], emotion_tensor)
            loss = criterion(output.reshape(-1, dataset.vocab_size), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
    return model, dataset

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt

# -------------------- Cornell Movie Dialogs Corpus'tan input-output çifti üretici --------------------
def load_cornell_data_json(save_path="cornell_data.json", max_pairs=10000):
    try:
        from convokit import Corpus, download
    except ImportError:
        print("convokit kütüphanesini yüklemeniz gerekiyor: pip install convokit")
        sys.exit(1)
    print("Cornell Movie Dialogs Corpus indiriliyor ve işleniyor...")
    corpus = Corpus(download("movie-corpus"))
    pairs = []
    for convo in corpus.iter_conversations():
        utts = convo.get_utterance_ids()
        for i in range(len(utts)-1):
            inp = corpus.get_utterance(utts[i]).text
            out = corpus.get_utterance(utts[i+1]).text
            if inp.strip() and out.strip():
                pairs.append({"input": inp, "output": out})
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"{len(pairs)} diyalog çifti {save_path} dosyasına kaydedildi.")
    return save_path

# -------------------- Main CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'chat'], help='Mode: train or chat')
    parser.add_argument('--data', type=str, help='Path to training data JSON veya \'cornell\' yazarsan Cornell veri seti kullanılır')
    parser.add_argument('--model_path', type=str, default='enhanced_brain_model.pt', help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Cornell veri seti seçildiyse otomatik indir ve dönüştür
    if args.data == "cornell":
        args.data = load_cornell_data_json(save_path="cornell_data.json", max_pairs=20000)

    if args.mode == 'train':
        if not args.data:
            print("Training mode requires --data argument with path to data.json")
            exit(1)
        print(f"Training on {device}...")
        model, dataset = train(args.data, args.epochs, args.batch_size, args.lr, device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'word2idx': dataset.word2idx,
            'idx2word': dataset.idx2word,
            'vocab_size': dataset.vocab_size
        }, args.model_path)
        print(f"Enhanced model saved to {args.model_path}")

    elif args.mode == 'chat':
        if not os.path.exists(args.model_path):
            print(f"Model file {args.model_path} not found! Train first.")
            exit(1)
        print("Loading model...")
        checkpoint = torch.load(args.model_path, map_location=device)
        model = BrainAgent(checkpoint['vocab_size']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        dataset = type('Dataset', (), {})()
        dataset.word2idx = checkpoint['word2idx']
        dataset.idx2word = checkpoint['idx2word']
        dataset.vocab_size = checkpoint['vocab_size']
        chat = BrainChat(model, dataset, device)
        print("Gelişmiş Sohbet Botu Aktif!")
        print("Komutlar:")
        print("/emotion <duygu> - Duygu modunu değiştir (happy, sad, angry, excited, neutral)")
        print("/correct <yanlış>|<doğru> - Botu düzeltme ile eğit")
        print("/insights - Botun kendi performans analizini gör")
        print("/exit - Çıkış yap")
        while True:
            user_inp = input("Sen: ").strip()
            if user_inp == "/exit":
                break
            response = chat.respond(user_inp)
            print("Bot:", response)
    torch.cuda.empty_cache()