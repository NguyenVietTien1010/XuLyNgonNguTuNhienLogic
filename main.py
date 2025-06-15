import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# ========== CẤU HÌNH ỨNG DỤNG ==========
st.set_page_config(page_title="Trợ Lý Logic Thời Tiết", layout="centered")

# ========== TẢI DỮ LIỆU ==========
@st.cache_data
def load_data(path="data2.csv"):
    try:
        return pd.read_csv(path, quotechar='"')
    except Exception as e:
        st.error(f"❌ Không thể tải dữ liệu: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# ========== TIỀN XỬ LÝ ==========
# Lọc bỏ nhãn không cần thiết
valid_actions = ['đi chơi', 'nghỉ ngơi']
df = df[df['action'].isin(valid_actions)]

le_weather, le_mood, le_action = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['weather_enc'] = le_weather.fit_transform(df['weather'].astype(str))
df['mood_enc'] = le_mood.fit_transform(df['mood'].astype(str))
df['action_enc'] = le_action.fit_transform(df['action'].astype(str))

# Đặc trưng logic bổ sung
def is_weather_good(w):
    return int(w in ["nắng", "nắng nhẹ", "đẹp"])

def is_mood_positive(m):
    return int(m in ["muốn", "rất muốn", "vui", "hứng thú", "mong muốn"])

df['is_weather_good'] = df['weather'].apply(is_weather_good)
df['is_mood_positive'] = df['mood'].apply(is_mood_positive)
df['logic_and'] = df['is_weather_good'] & df['is_mood_positive']
df['logic_or'] = df['is_weather_good'] | df['is_mood_positive']
df['logic_xor'] = df['is_weather_good'] ^ df['is_mood_positive']

# Hiển thị nhãn

def show_labels(title, encoder):
    label_map = {i: label for i, label in enumerate(encoder.classes_)}
    st.markdown(f"### {title}")
    st.json(label_map)

# ========== CHUẨN BỊ DỮ LIỆU ==========
X = df[['weather_enc', 'mood_enc', 'is_weather_good', 'is_mood_positive', 'logic_and', 'logic_or', 'logic_xor']].values
y = df['action_enc'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ========== MÔ HÌNH MLP MỚI ==========
n_classes = df['action_enc'].nunique()

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
model_path = "mlp_model.pth"

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=np.array(y_train))
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model():
    for epoch in range(500):
        model.train()
        y_pred = model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                train_preds = model(X_train_tensor)
                acc_train = accuracy_score(y_train, torch.argmax(train_preds, dim=1).numpy())
                print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}, Train Accuracy = {acc_train*100:.2f}%")

# ========== HUẤN LUYỆN ==========
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except RuntimeError as e:
        st.warning(f"⚠️ Mô hình cũ không tương thích: {e}")
        st.info("🔄 Đang huấn luyện lại mô hình từ đầu...")
        train_model()
        torch.save(model.state_dict(), model_path)
else:
    train_model()
    torch.save(model.state_dict(), model_path)

# ========== ĐÁNH GIÁ ==========
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    predicted = torch.argmax(y_pred_test, dim=1)
    acc = accuracy_score(y_test, predicted.numpy())

# ========== TRỢ LÝ DỰ ĐOÁN ==========
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
tab1, tab2, tab3 = st.tabs(["🧫 Trợ lý dự đoán", "📊 Thống kê & dữ liệu", "🌈 Biểu diễn MLP"])

with tab1:
    st.title("🤖 Trợ lý Logic – Dự đoán hành động từ thời tiết & cảm xúc")

    st.markdown("""
    🔍 **Hướng dẫn nhập liệu:**

    Vui lòng sử dụng các từ khóa hợp lệ sau trong phần mô tả:

    - **Thời tiết**: `nắng`, `gió`, `gió to`, `mưa`, `âm u`, `nhiều mây`
    - **Tâm trạng**: `muốn`, `rất muốn`, `không muốn`, `thèm`, `mong muốn`, `vui`, `buồn`, `hứng thú`, `mệt`, `khó chịu`, `...`

    👉 Ví dụ: *"Hôm nay trời nắng và tôi rất muốn đi ra ngoài"*
    """)

user_input = st.text_input("📝 Hãy nhập tình huống (VD: 'Trời mưa và tôi muốn đi chơi'):")

def extract_features(text):
    weather_map = ["gió to","mưa phùn","mưa rào", "mưa","nắng gắt","nắng nhẹ", "nắng", "gió", "âm u", "nhiều mây", "trời sấm", "bão", "đẹp", "lạnh"]
    mood_map = ["rất muốn", "không muốn", "muốn", "thèm", "mong muốn", "vui", "buồn", "hứng thú", "mệt", "khó chịu", "chán","kiệt sức","đầy năng lượng"]
    found_weather = next((w for w in weather_map if w in text.lower()), None)
    found_mood = next((m for m in mood_map if m in text.lower()), None)
    return found_weather, found_mood

def apply_logic_rules(weather, mood):
    A = 1 if weather in ["nắng"] else 0
    B = 1 if mood in ["muốn", "rất muốn","vui"] else 0 
    and_result = A & B
    or_result = A | B
    not_a = int(not A)
    xor_result = A ^ B

    and_exp = "✅ Thời tiết đẹp và bạn muốn đi → đi chơi là hợp lý." if and_result else "❌ Một trong hai điều kiện không thỏa mãn → không nên đi."
    or_exp = "✅ Ít nhất một điều kiện phù hợp → có thể cân nhắc đi." if or_result else "❌ Cả hai điều kiện đều không tốt → nên ở nhà."
    not_exp = "✅ Thời tiết không đẹp → nên nghỉ ngơi." if not_a else "ℹ️ Thời tiết đẹp."
    xor_exp = "✅ Chỉ một điều kiện đúng → nên cân nhắc." if xor_result else "❌ Cả hai điều kiện đều giống nhau → cần suy nghĩ thêm."

    chat_decision = "✅ Kết luận: Bạn nên đi chơi hôm nay!" if and_result or (xor_result and A == 1) else "🛋️ Kết luận: Bạn nên nghỉ ngơi thì hơn."

    result_logic = f"""
🔢 **Biểu diễn logic:**

- A (thời tiết đẹp) = {A}  
- B (muốn đi) = {B}  
- **AND (A ∧ B)** = {and_result} → {and_exp}  
- **OR (A ∨ B)** = {or_result} → {or_exp}  
- **NOT A (¬A)** = {not_a} → {not_exp}  
- **XOR (A ⊕ B)** = {xor_result} → {xor_exp}  
"""
    return result_logic, chat_decision

def get_explanation(weather, mood, action_idx):
    row = df[(df['weather'] == weather) & (df['mood'] == mood)]
    if not row.empty:
        return row.iloc[0]['explanation']
    return {
        0: f"Trời {weather}, tâm trạng {mood} → nên nghỉ ngơi là hợp lý.",
        1: f"Trời {weather}, tâm trạng {mood} → đi ra ngoài sẽ thú vị đấy!"
    }.get(action_idx, "🤔 Tôi chưa chắc lắm trong tình huống này.")

if user_input:
    weather, mood = extract_features(user_input)
    st.write(f"📌 Trích xuất: `weather = {weather}`, `mood = {mood}`")

    if weather in le_weather.classes_ and mood in le_mood.classes_:
        try:
            w_enc = le_weather.transform([weather])[0]
            m_enc = le_mood.transform([mood])[0]
            x_input = torch.tensor([[w_enc, m_enc]], dtype=torch.float32)

            with torch.no_grad():
                output = model(x_input)
                pred_idx = torch.argmax(output, dim=1).item()
                probability = torch.softmax(output, dim=1)[0, pred_idx].item()
                action_label = le_action.inverse_transform([pred_idx])[0]

            explanation = get_explanation(weather, mood, pred_idx)
            logic_output, chat_decision = apply_logic_rules(weather, mood)

            st.success(f"""
                🌤️ Thời tiết: **{weather}** → `{w_enc}`  
                💭 Tâm trạng: **{mood}** → `{m_enc}`  
                🤖 Dự đoán (MLP): **{action_label}** (xác suất: `{probability:.2f}`)  
                🧠 Giải thích theo dữ liệu: {explanation}
            """)

            st.error(logic_output)

            st.session_state.chat_history.append({"user": user_input, "bot": chat_decision})

            with st.chat_message("assistant"):
                st.markdown(chat_decision)

            if weather == "nắng":
                st.balloons()
            elif weather == "âm u" or weather == "mưa" or weather == "nhiều mây":
                st.snow()

            st.divider()
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(chat["user"])
                with st.chat_message("assistant"):
                    st.markdown(chat["bot"])
        except Exception as e:
            st.error(f"❌ Lỗi xử lý: {e}")

        
    else:
        st.warning(f"""
            ⚠️ Không nhận diện được đúng từ khóa:  
            - Thời tiết: `{weather}` {'✅' if weather in le_weather.classes_ else '❌ không hợp lệ'}`  
            - Tâm trạng: `{mood}` {'✅' if mood in le_mood.classes_ else '❌ không hợp lệ'}`  
            💡 Vui lòng sử dụng đúng từ khóa từ bảng nhãn phía trên.
        """)


with tab2:
    st.markdown("### 📊 Thống kê mô hình")
    st.markdown(f"🌟 **Độ chính xác mô hình:** `{acc*100:.2f}%`")

    # Biểu đồ phân bố hành động
    st.markdown("### 📈 Biểu đồ phân bố hành động")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='action', order=df['action'].value_counts().index, palette='Set2', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Tần suất các hành động được dự đoán")
    ax.set_xlabel("Hành động")
    ax.set_ylabel("Số lượng")
    st.pyplot(fig)

    st.markdown("### 📋 Bảng dữ liệu gốc")
    st.dataframe(df)

    st.markdown("### 🌿 Nhãn đã mã hóa")
    show_labels("🌦️ Nhãn thời tiết", le_weather)
    show_labels("🧠 Nhãn tâm trạng", le_mood)
    show_labels("🏃‍♀️ Nhãn hành động", le_action)

with tab3:
    import graphviz as gv

    def draw_mlp():
        g = gv.Digraph(format="png")
        g.attr(rankdir='LR', size='10')

        # Lớp đầu vào
        g.node("Input\n(weather, mood)", shape="circle", style="filled", color="lightblue")

        # Ẩn lớp 1
        for i in range(1, 5):
            g.node(f"h1_{i}", f"H1-{i}", shape="circle", style="filled", color="lightgreen")
            g.edge("Input\n(weather, mood)", f"h1_{i}")

        # Ẩn lớp 2
        for j in range(1, 3):
            g.node(f"h2_{j}", f"H2-{j}", shape="circle", style="filled", color="palegreen")
            for i in range(1, 5):
                g.edge(f"h1_{i}", f"h2_{j}")

        # Lớp đầu ra
        for k in range(n_classes):
            g.node(f"out_{k}", f"Output {k}", shape="doublecircle", style="filled", color="gold")
            for j in range(1, 3):
                g.edge(f"h2_{j}", f"out_{k}")

        return g

    st.graphviz_chart(draw_mlp())
