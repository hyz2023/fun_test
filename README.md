# 说话人识别演示

这个项目使用SpeechBrain库实现了说话人识别功能。

## 功能特点

- 使用预训练的SpeechBrain模型进行说话人识别
- 支持音频文件上传进行说话人识别
- 支持实时麦克风录音进行说话人识别
- 简单的Web界面用于演示

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 安装依赖
2. 运行Web应用

```bash
python app.py
```

3. 在浏览器中打开显示的URL
4. 上传音频文件或使用麦克风录制声音进行说话人识别

## 项目结构

- `app.py`: Web应用主程序
- `speaker_recognition.py`: 说话人识别模型实现
- `templates/`: Web界面模板
- `static/`: 静态资源文件
- `requirements.txt`: 项目依赖