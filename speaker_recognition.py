import os
import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from pydub import AudioSegment
import io
# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 在导入plt之前设置后端
import matplotlib.pyplot as plt
import librosa
import librosa.display

class SpeakerRecognition:
    def __init__(self):
        # 加载预训练的说话人识别模型
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        # 存储已注册的说话人嵌入向量
        self.speaker_embeddings = {}
        # 存储说话人音频文件路径
        self.audio_paths = {}
        
    def save_audio_path(self, name, audio_path):
        """保存说话人音频文件路径"""
        self.audio_paths[name] = audio_path
        return True
        
    def get_audio_path(self, name):
        """获取说话人音频文件路径"""
        return self.audio_paths.get(name)
        
    def process_audio(self, audio_file):
        """处理音频文件，返回波形和采样率"""
        if isinstance(audio_file, str):
            # 如果是文件路径
            signal, fs = torchaudio.load(audio_file)
        else:
            # 如果是上传的文件或字节流
            try:
                # 尝试直接加载
                signal, fs = torchaudio.load(audio_file)
            except Exception:
                # 如果失败，尝试通过pydub处理
                audio_data = AudioSegment.from_file(audio_file)
                audio_array = np.array(audio_data.get_array_of_samples())
                
                # 转换为float32并归一化
                if audio_data.sample_width == 2:  # 16-bit audio
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif audio_data.sample_width == 4:  # 32-bit audio
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                
                # 转换为PyTorch张量
                signal = torch.FloatTensor(audio_array).unsqueeze(0)
                fs = audio_data.frame_rate
        
        # 确保音频是单声道
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # 确保采样率为16kHz（SpeechBrain模型的要求）
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
            fs = 16000
            
        return signal, fs
    
    def extract_embedding(self, audio_file):
        """从音频文件中提取说话人嵌入向量"""
        signal, fs = self.process_audio(audio_file)
        embedding = self.model.encode_batch(signal)
        return embedding.squeeze().cpu().detach().numpy()
    
    def register_speaker(self, name, audio_file):
        """注册新说话人"""
        embedding = self.extract_embedding(audio_file)
        self.speaker_embeddings[name] = embedding
        return True
    
    def identify_speaker(self, audio_file, threshold=0.75):
        """识别说话人身份"""
        if not self.speaker_embeddings:
            return {"error": "没有注册的说话人"}
        
        # 提取测试音频的嵌入向量
        test_embedding = self.extract_embedding(audio_file)
        
        # 计算与所有注册说话人的余弦相似度
        similarities = {}
        for name, embedding in self.speaker_embeddings.items():
            similarity = self._compute_similarity(test_embedding, embedding)
            similarities[name] = float(similarity)
        
        # 找出最相似的说话人
        best_match = max(similarities.items(), key=lambda x: x[1])
        
        # 如果相似度低于阈值，则认为是未知说话人
        if best_match[1] < threshold:
            result = {
                "identity": "未知说话人",
                "confidence": 0.0,
                "similarities": similarities
            }
        else:
            result = {
                "identity": best_match[0],
                "confidence": best_match[1],
                "similarities": similarities
            }
            
        return result
    
    def _compute_similarity(self, embedding1, embedding2):
        """计算两个嵌入向量之间的余弦相似度"""
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return np.dot(embedding1, embedding2)
    
    def get_registered_speakers(self):
        """获取所有已注册的说话人"""
        return list(self.speaker_embeddings.keys())
    
    def remove_speaker(self, name):
        """删除已注册的说话人"""
        if name in self.speaker_embeddings:
            # 删除嵌入向量
            del self.speaker_embeddings[name]
            # 同时删除音频路径信息
            if name in self.audio_paths:
                del self.audio_paths[name]
            return True
        return False
    
    def save_speakers(self, file_path="speakers.npz"):
        """保存已注册的说话人数据和音频路径"""
        # 创建一个字典，包含嵌入向量和音频路径
        data_to_save = {}
        
        # 保存嵌入向量，使用'emb_'前缀区分
        for name, embedding in self.speaker_embeddings.items():
            data_to_save[f'emb_{name}'] = embedding
        
        # 保存音频路径，使用'path_'前缀区分，将路径转换为字符串数组
        for name, path in self.audio_paths.items():
            # 将路径转换为UTF-8字符串的字节数组
            path_bytes = np.array([ord(c) for c in path], dtype=np.uint8)
            data_to_save[f'path_{name}'] = path_bytes
        
        np.savez(file_path, **data_to_save)
        return True
    
    def load_speakers(self, file_path="speakers.npz"):
        """加载已注册的说话人数据和音频路径"""
        if os.path.exists(file_path):
            data = np.load(file_path)
            
            # 清空现有数据
            self.speaker_embeddings = {}
            self.audio_paths = {}
            
            # 处理所有保存的数据
            for key in data.files:
                if key.startswith('emb_'):
                    # 提取说话人名称（去掉'emb_'前缀）
                    name = key[4:]
                    self.speaker_embeddings[name] = data[key]
                elif key.startswith('path_'):
                    # 提取说话人名称（去掉'path_'前缀）
                    name = key[5:]
                    # 将字节数组转换回字符串
                    path_bytes = data[key]
                    path = ''.join(chr(b) for b in path_bytes)
                    self.audio_paths[name] = path
            
            return True
        return False
    
    def visualize_waveform(self, audio_file):
        """可视化音频波形"""
        signal, fs = self.process_audio(audio_file)
        signal_np = signal.squeeze().numpy()
        
        # 使用非交互式后端，避免在非主线程创建GUI窗口
        import matplotlib
        matplotlib.use('Agg')  # 使用Agg后端，不需要GUI
        
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, len(signal_np) / fs, len(signal_np)), signal_np)
        plt.title('音频波形')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.tight_layout()
        
        # 保存到内存中的字节流
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def visualize_spectrogram(self, audio_file):
        """可视化音频频谱图"""
        signal, fs = self.process_audio(audio_file)
        signal_np = signal.squeeze().numpy()
        
        # 使用非交互式后端，避免在非主线程创建GUI窗口
        import matplotlib
        matplotlib.use('Agg')  # 使用Agg后端，不需要GUI
        
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal_np)), ref=np.max)
        librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('频谱图')
        plt.tight_layout()
        
        # 保存到内存中的字节流
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf