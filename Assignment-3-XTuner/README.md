# InternLM-Camp-2024
第四课作业：

基础作业：
在本地、Gradio 上或通过 API 部署 InternLM-Chat-7B，并生成 300 个字符的故事
在 OpenXLab 上部署作业 3（已完成）
InterLM-Chat-7B 使用 KV Cache 进行模型量化并通过 API 进行部署
通过 TurboMind Python 与 W4A116、KV 缓存或 W4A16 + KV 缓存对自定义数据进行基准比较。此外，尝试使用 HuggingFace 进行推理

安装LMDeploy
在InternStudioCuda11.7-conda上使用docker 映像启动 A100 (1/4) 运行时

在终端中创建新环境（bash）： conda create --name lmdeploy --clone=/root/share/conda_envs/internlm-base

激活新环境后conda activate lmdeploy，我们需要安装lmdeploy：
```
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl # precompiled flash attention
pip install cmake lit packaging 'lmdeploy[all]==v0.1.0'
```
本地加载模型

```
lmdeploy chat turbomind /share/temp/model_repos/internlm-chat-7b/  --model-name internlm-chat-7b
```

RAM消耗为14854MiB

```
+------------------------------------------------------------------------------+
| VGPU-SMI 1.7.13       Driver Version: 535.54.03     CUDA Version: 12.2       |
+-------------------------------------------+----------------------------------+
| GPU  Name                Bus-Id           |        Memory-Usage     GPU-Util |
|===========================================+==================================|
|   0  NVIDIA A100-SXM...  00000000:19:00.0 | 14854MiB / 20470MiB    0% /  25% |
+-------------------------------------------+----------------------------------+
```
通过 TurboMind 部署

```
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/
```

通过 TurboMind 启动模型
```
lmdeploy chat turbomind ./workspace
```
RAM消耗是14750MiB

```
+------------------------------------------------------------------------------+
| VGPU-SMI 1.7.13       Driver Version: 535.54.03     CUDA Version: 12.2       |
+-------------------------------------------+----------------------------------+
| GPU  Name                Bus-Id           |        Memory-Usage     GPU-Util |
|===========================================+==================================|
|   0  NVIDIA A100-SXM...  00000000:19:00.0 | 14750MiB / 20470MiB    0% /  25% |
+-------------------------------------------+----------------------------------+
```
通过 TurboMind + API 进行部署
使用 API 启动 TurboMind


```
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1
```
访问API


```
lmdeploy serve api_client http://localhost:23333
```
RAM消耗为14790MiB

```
+------------------------------------------------------------------------------+
| VGPU-SMI 1.7.13       Driver Version: 535.54.03     CUDA Version: 12.2       |
+-------------------------------------------+----------------------------------+
| GPU  Name                Bus-Id           |        Memory-Usage     GPU-Util |
|===========================================+==================================|
|   0  NVIDIA A100-SXM...  00000000:19:00.0 | 14790MiB / 20470MiB    0% /  25% |
+-------------------------------------------+----------------------------------+
```
通过 SSH 连接服务器

```
ssh -CNg -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 37052
```

![pic1](asserts/img1.png)

在 127.0.0.1:23333 上访问它 生成一个故事/v1/chat/completinos

![pic2](asserts/img2.png)



```
"content": "。杭州，自古以来就是繁华之地，是江南的代表性城市。西湖，是杭州最具有代表性的景点之一。西湖的美丽，不仅在于她的自然风光，更在于她的文化底蕴。西湖的美，不仅仅在于她的山水，更在于她的历史与人文。西湖的美，不仅仅在于她的风景，更在于她的故事。西湖的故事，不仅仅在于她的传说，更在于她的现实。西湖的故事，不仅仅在于她的文化，更在于她的传承。西湖的故事，不仅仅在于她的历史，更在于她的未来。西湖的故事，不仅仅在于她的现在，更在于她的明天。西湖的故事，不仅仅在于她的中国，更在于她的世界。西湖的故事，不仅仅在于她的传说，更在于她的现实。西湖的故事，不仅仅在于她的文化，更在于她的传承。西湖的故事，不仅仅在于她的历史，更在于她的未来。西湖的故事，不仅仅在于她的现在，更在于她的明天。西湖的故事，不仅仅在于她的中国，更在于她的世界。西湖的故事，不仅仅在于她的传说，更在于她的现实。西湖的故事，不仅仅在于她的文化，更在于她的传承。西湖的故事，不仅仅在于她的历史，更在于她的未来。西湖的故事，不仅仅在于她的现在，更在于她的明天。西湖的故事，不仅仅在于她的中国，更在于她的世界。西湖的故事，不仅仅在于她的传说，更在于她的现实。西湖的故事，不仅仅在于她的文化，更在于她的传承。西湖的故事，不仅仅在于她的历史，更在于她的未来。西湖的故事，不仅仅在于她的现在，更在于她的明天。西湖的故事，不仅仅在于她的中国，更在于她的世界。西湖的故事，不仅仅在于她的传说，更在于她的现实。西湖的故事，不仅仅在于她的文化，更在于她的传承。西湖的故事，不仅仅在于她的历史，更在于她的未来。西湖的故事，不仅仅在于她的现在，更在于她的明天。西湖的故事，不仅仅在于她的中国，更在于她的世界。西湖的故事，不仅仅在于她的传说，更在于她的现实。西湖的故事，不仅仅在于她的文化，更在于她的传承。西湖的故事，不仅仅在于她的历史，更在于她的未来。西湖的故事，不仅仅在于她的现在，更在于她的明天。西湖的故事，不仅仅在于她的中国，更在于她的世界。西湖的故事，不仅仅在于她的传说，更在于她的现实。西湖的故事，不仅仅在于她的文化，更在于她的传承。西湖的故事，不仅仅在于她的历史，更在于她的未来。西湖的故事，不仅仅在于她的现在，更在于她的明天。西湖的故事"
```

通过 TurboMind +API + Gradio 部署（后端：API）
要求：首先像上一节一样启动服务器


```
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True
```
也可以看到本地机器的端口 6006

```
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 37052
```
在 127.0.0.1:6006 上访问

通过 TurboMind + Gradio 部署（后端：TurboMind）
无服务器要求

```
lmdeploy serve gradio ./workspace
```

通过 SSH 连接服务器

```
(lmdeploy) root@intern-studio:~# lmdeploy serve gradio ./workspace
/root/.conda/envs/lmdeploy/lib/python3.10/site-packages/gradio/components/button.py:89: UserWarning: Using the update method is deprecated. Simply return a new object instead, e.g. `return gr.Button(...)` instead of `return gr.Button.update(...)`.
  warnings.warn(
model_source: workspace
WARNING: Can not find tokenizer.json. It may take long time to initialize the tokenizer.
[TM][WARNING] [LlamaTritonModel] `max_context_token_num` = 2056.
[TM][WARNING] [LlamaTritonModel] `num_tokens_per_iter` is not set, default to `max_context_token_num` (2056).
[WARNING] gemm_config.in is not found; using default GEMM algo
[TM][INFO] NCCL group_id = 0
[TM][INFO] [BlockManager] block_size = 64 MB
[TM][INFO] [BlockManager] max_block_count = 159
[TM][INFO] [BlockManager] chunk_size = 1
[TM][INFO] LlamaBatch<T>::Start()
server is gonna mount on: http://0.0.0.0:6006
Running on local URL:  http://0.0.0.0:6006
```

C4 数据集上的 InterLM-Chat-7B 模型量化
方法

KV Cache INT8：令牌键和值 fp16 -> int8：从默认 0（禁用）修改quant_policy为 4（启用）
W4A16：权重量化为 4 位，而激活保留在 fp16 中。
Linear Rope Scaling 和 Dynamic NTK Scaling：允许比训练更长的文本输入：rope_scaling_factor从 0.0 修改为 1.0 以及use_logn_attn从 0 修改为 1
批量槽：修改max_batch_size又名instance_num

**准备数据集** ``` python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple cp /root/share/temp/datasets/c4/calib_dataloader。 py /root/.conda/envs/lmdeploy/lib/python3.10/site-packages/lmdeploy/lite/utils/ cp -r /root/share/temp/datasets/c4/ /root/.cache/huggingface/datasets /```
KV Cache INT8 计算minmax（每个模型层上输入的大小）并用它来校准样本数量samples和样本长度seqln

```
lmdeploy lite calibrate \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --calib_dataset "c4" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_output
```


```
Loading calibrate dataset ...
model.layers.0, samples: 128, max gpu memory: 8.18 GB
model.layers.1, samples: 128, max gpu memory: 10.18 GB
model.layers.2, samples: 128, max gpu memory: 10.18 GB
model.layers.3, samples: 128, max gpu memory: 10.18 GB
model.layers.4, samples: 128, max gpu memory: 10.18 GB
model.layers.5, samples: 128, max gpu memory: 10.18 GB
model.layers.6, samples: 128, max gpu memory: 10.18 GB
model.layers.7, samples: 128, max gpu memory: 10.18 GB
model.layers.8, samples: 128, max gpu memory: 10.18 GB
model.layers.9, samples: 128, max gpu memory: 10.18 GB
model.layers.10, samples: 128, max gpu memory: 10.18 GB
model.layers.11, samples: 128, max gpu memory: 10.18 GB
model.layers.12, samples: 128, max gpu memory: 10.18 GB
model.layers.13, samples: 128, max gpu memory: 10.18 GB
model.layers.14, samples: 128, max gpu memory: 10.18 GB
model.layers.15, samples: 128, max gpu memory: 10.18 GB
model.layers.16, samples: 128, max gpu memory: 10.18 GB
model.layers.17, samples: 128, max gpu memory: 10.18 GB
model.layers.18, samples: 128, max gpu memory: 10.18 GB
model.layers.19, samples: 128, max gpu memory: 10.18 GB
model.layers.20, samples: 128, max gpu memory: 10.18 GB
model.layers.21, samples: 128, max gpu memory: 10.18 GB
model.layers.22, samples: 128, max gpu memory: 10.18 GB
model.layers.23, samples: 128, max gpu memory: 10.18 GB
model.layers.24, samples: 128, max gpu memory: 10.18 GB
model.layers.25, samples: 128, max gpu memory: 10.18 GB
model.layers.26, samples: 128, max gpu memory: 10.18 GB
model.layers.27, samples: 128, max gpu memory: 10.18 GB
model.layers.28, samples: 128, max gpu memory: 10.18 GB
model.layers.29, samples: 128, max gpu memory: 10.18 GB
model.layers.30, samples: 128, max gpu memory: 10.18 GB
model.layers.31, samples: 128, max gpu memory: 10.18 GB
```
计算量化参数

```
lmdeploy lite kv_qparams \
  --work_dir ./quant_output  \
  --turbomind_dir workspace/triton_models/weights/ \
  --kv_sym False \
  --num_tp 1
```

```
Layer 0 MP 0 qparam:    0.06256103515625        -0.626953125    0.00891876220703125     0.029296875
Layer 1 MP 0 qparam:    0.0892333984375         -1.2109375      0.0207672119140625      -0.1923828125
Layer 2 MP 0 qparam:    0.0771484375    -0.015625       0.03631591796875        -0.8349609375
Layer 3 MP 0 qparam:    0.09783935546875        0.0390625       0.0296173095703125      -0.955078125
Layer 4 MP 0 qparam:    0.11334228515625        0.0625  0.03228759765625        0.244140625
Layer 5 MP 0 qparam:    0.1180419921875         0.16015625      0.050994873046875       1.251953125
Layer 6 MP 0 qparam:    0.1226806640625         -0.00390625     0.0306549072265625      0.275390625
Layer 7 MP 0 qparam:    0.134033203125  -0.7421875      0.03826904296875        -0.2265625
Layer 8 MP 0 qparam:    0.135009765625  -0.25   0.03753662109375        -1.0791015625
Layer 9 MP 0 qparam:    0.136474609375  -0.078125       0.037567138671875       0.544921875
Layer 10 MP 0 qparam:   0.1302490234375         -1.39453125     0.0265960693359375      -0.0458984375
Layer 11 MP 0 qparam:   0.1317138671875         0.359375        0.03216552734375        0.2529296875
Layer 12 MP 0 qparam:   0.131103515625  0.2734375       0.030731201171875       -0.0654296875
Layer 13 MP 0 qparam:   0.1357421875    0.1640625       0.0350341796875         0.044921875
Layer 14 MP 0 qparam:   0.127197265625  -0.2734375      0.030731201171875       0.0546875
Layer 15 MP 0 qparam:   0.131103515625  0.0703125       0.0345458984375         -0.40625
Layer 16 MP 0 qparam:   0.13232421875   -1.1015625      0.038177490234375       0.373046875
Layer 17 MP 0 qparam:   0.1287841796875         0.70703125      0.04364013671875        -0.48046875
Layer 18 MP 0 qparam:   0.1270751953125         -0.28125        0.04669189453125        -0.0390625
Layer 19 MP 0 qparam:   0.13916015625   -0.0703125      0.053680419921875       -0.39453125
Layer 20 MP 0 qparam:   0.154296875     -0.796875       0.04779052734375        -0.1171875
Layer 21 MP 0 qparam:   0.134521484375  0.0703125       0.051025390625  -1.17578125
Layer 22 MP 0 qparam:   0.1405029296875         -0.28125        0.053131103515625       -0.345703125
Layer 23 MP 0 qparam:   0.130615234375  -0.015625       0.0941162109375         -0.41796875
Layer 24 MP 0 qparam:   0.125732421875  -0.3203125      0.048980712890625       0.16015625
Layer 25 MP 0 qparam:   0.1390380859375         0.015625        0.06494140625   -0.125
Layer 26 MP 0 qparam:   0.1328125       -0.8828125      0.087646484375  -0.15234375
Layer 27 MP 0 qparam:   0.13427734375   -0.1015625      0.066650390625  0.1484375
Layer 28 MP 0 qparam:   0.1416015625    0.25    0.07989501953125        -0.6484375
Layer 29 MP 0 qparam:   0.1405029296875         0.484375        0.0819091796875         0.984375
Layer 30 MP 0 qparam:   0.135009765625  -0.5546875      0.0869140625    -0.35546875
Layer 31 MP 0 qparam:   0.133056640625  0.7109375       0.1517333984375         -0.25
```

W4A16 将权重转换为torch.int4

```
lmdeploy lite auto_awq \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_output
```

```
...
model.layers.30.self_attn.q_proj weight packed.
model.layers.30.self_attn.k_proj weight packed.
model.layers.30.self_attn.v_proj weight packed.
model.layers.30.self_attn.o_proj weight packed.
model.layers.30.mlp.gate_proj weight packed.
model.layers.30.mlp.down_proj weight packed.
model.layers.30.mlp.up_proj weight packed.
model.layers.31.self_attn.q_proj weight packed.
model.layers.31.self_attn.k_proj weight packed.
model.layers.31.self_attn.v_proj weight packed.
model.layers.31.self_attn.o_proj weight packed.
model.layers.31.mlp.gate_proj weight packed.
model.layers.31.mlp.down_proj weight packed.
model.layers.31.mlp.up_proj weight packed.
```

转换模型布局

```
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128 \
    --dst_path ./workspace_quant
```


```...
### copying layers.30.attention.wo.bias, shape=torch.Size([4096])                                                                                                                                                                           
*** splitting layers.30.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                                                                                                  
*** splitting layers.30.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                                                                                                    
*** splitting layers.30.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                                                                                          
*** splitting layers.30.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                                                                                                       
*** splitting layers.31.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                                                                                                   
*** splitting layers.31.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                                                                                                     
*** splitting layers.31.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                                                                                              
*** splitting layers.31.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                                                                                          
*** splitting layers.31.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                                                                                              
### copying layers.31.attention.wo.bias, shape=torch.Size([4096])                                                                                                                                                                           
*** splitting layers.31.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                                                                                                  
*** splitting layers.31.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                                                                                                    
*** splitting layers.31.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                                                                                          
*** splitting layers.31.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1 

```



































