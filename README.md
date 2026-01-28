# CDAN
## 项目部署
在新的python环境中执行
```
pip install -r requirements.txt
```
然后在Huggingface上下载对应模型的文件，并且保存在本地对应的文件夹中  
修改代码中CONFIG对应的路径，确保数据文件和模型文件加载正确    
需要下载的模型如下：  
https://huggingface.co/openai/clip-vit-base-patch32/tree/main  
https://huggingface.co/google/vit-base-patch16-224/tree/main  
https://huggingface.co/google-bert/bert-base-uncased/tree/main  

## 文件说明
cdan_model.py  ==>  模型定义文件  
data_preprocess.py  ==>  数据预处理相关函数  
train.py  ==>  训练+测试代码  
save_val_guids.py  ==>  保存验证集数据文件  
