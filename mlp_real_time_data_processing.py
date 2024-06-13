import numpy as np

# 假設我們有20個channel的實時數據
num_channels = 20
num_samples = 1000  # 總樣本數

# 生成隨機的即時數據
data = np.random.random((num_samples, num_channels))

# 假設我們有每10個樣本形成一個滑動窗口
window_size = 10
stride = 5  # 滑動窗口的步長

# 提取特徵
def extract_features(data, window_size, stride):
    num_windows = (len(data) - window_size) // stride + 1
    features = []
    for i in range(0, num_windows * stride, stride):
        window = data[i:i+window_size]
        # 提取窗口內的特徵，例如平均值
        mean_features = np.mean(window, axis=0)
        max_features = np.max(window, axis=0)
        min_features = np.min(window, axis=0)
        std_features = np.std(window, axis=0)
        features.append(np.concatenate([mean_features, max_features, min_features, std_features]))
    return np.array(features)

x_train = extract_features(data, window_size, stride)

# 為簡單起見，我們隨機生成標籤
num_features = x_train.shape[1]
y_train = np.random.randint(2, size=len(x_train))

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_dim=num_features),
    tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=20, batch_size=128)

# 測試數據
x_test = extract_features(np.random.random((100, num_channels)), window_size, stride)
y_test = np.random.randint(2, size=len(x_test))

# 評估模型
score = model.evaluate(x_test, y_test, batch_size=128)
print("score:", score)

# 預測
predict = model.predict(x_test)
print("predict:", predict)

# 輸出每個樣本的預測類別
predicted_classes = np.argmax(predict, axis=1)
print("Predicted classes:", predicted_classes)
print("y_test", y_test[:])
