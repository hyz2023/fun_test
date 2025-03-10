from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
from werkzeug.utils import secure_filename
from speaker_recognition import SpeakerRecognition
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化说话人识别模型
speaker_recognition = SpeakerRecognition()

# 尝试加载已保存的说话人数据
speaker_recognition.load_speakers()

@app.route('/')
def index():
    speakers = speaker_recognition.get_registered_speakers()
    return render_template('index.html', speakers=speakers)

@app.route('/register', methods=['POST'])
def register_speaker():
    if 'audio' not in request.files or 'name' not in request.form:
        return jsonify({'error': '缺少必要参数'}), 400
    
    audio_file = request.files['audio']
    name = request.form['name']
    
    if audio_file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 保存上传的音频文件
    filename = secure_filename(f"{name}_{audio_file.filename}")
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(audio_path)
    
    try:
        # 注册说话人
        speaker_recognition.register_speaker(name, audio_path)
        # 保存音频文件路径
        speaker_recognition.save_audio_path(name, audio_path)
        # 保存说话人数据（包括嵌入向量和音频路径）
        speaker_recognition.save_speakers()
        
        # 生成波形图和频谱图
        waveform = speaker_recognition.visualize_waveform(audio_path)
        spectrogram = speaker_recognition.visualize_spectrogram(audio_path)
        
        # 将图像转换为base64编码
        waveform_b64 = base64.b64encode(waveform.getvalue()).decode('utf-8')
        spectrogram_b64 = base64.b64encode(spectrogram.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'message': f'成功注册说话人: {name}',
            'waveform': waveform_b64,
            'spectrogram': spectrogram_b64
        })
    except Exception as e:
        # 如果注册失败，清理已上传的音频文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'error': f'注册失败: {str(e)}'}), 500

@app.route('/identify', methods=['POST'])
def identify_speaker():
    if 'audio' not in request.files:
        return jsonify({'error': '缺少音频文件'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 保存上传的音频文件
    filename = secure_filename(audio_file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(temp_path)
    
    try:
        # 识别说话人
        result = speaker_recognition.identify_speaker(temp_path)
        
        # 生成波形图和频谱图
        waveform = speaker_recognition.visualize_waveform(temp_path)
        spectrogram = speaker_recognition.visualize_spectrogram(temp_path)
        
        # 将图像转换为base64编码
        waveform_b64 = base64.b64encode(waveform.getvalue()).decode('utf-8')
        spectrogram_b64 = base64.b64encode(spectrogram.getvalue()).decode('utf-8')
        
        # 添加图像到结果
        result['waveform'] = waveform_b64
        result['spectrogram'] = spectrogram_b64
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'识别失败: {str(e)}'}), 500
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/speakers', methods=['GET'])
def get_speakers():
    speakers = speaker_recognition.get_registered_speakers()
    return jsonify({'speakers': speakers})

@app.route('/speaker_visualization/<name>', methods=['GET'])
def get_speaker_visualization(name):
    # 检查说话人是否存在
    speakers = speaker_recognition.get_registered_speakers()
    if name not in speakers:
        return jsonify({'error': f'未找到说话人: {name}'}), 404
    
    try:
        # 获取说话人的音频文件路径
        audio_path = speaker_recognition.get_audio_path(name)
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': f'未找到说话人的音频文件'}), 404
        
        # 生成波形图和频谱图
        waveform = speaker_recognition.visualize_waveform(audio_path)
        spectrogram = speaker_recognition.visualize_spectrogram(audio_path)
        
        # 将图像转换为base64编码
        waveform_b64 = base64.b64encode(waveform.getvalue()).decode('utf-8')
        spectrogram_b64 = base64.b64encode(spectrogram.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'speaker': name,
            'waveform': waveform_b64,
            'spectrogram': spectrogram_b64
        })
    except Exception as e:
        return jsonify({'error': f'获取可视化失败: {str(e)}'}), 500

@app.route('/remove_speaker/<name>', methods=['DELETE'])
def remove_speaker(name):
    success = speaker_recognition.remove_speaker(name)
    if success:
        speaker_recognition.save_speakers()
        return jsonify({'success': True, 'message': f'成功删除说话人: {name}'})
    else:
        return jsonify({'error': f'未找到说话人: {name}'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)