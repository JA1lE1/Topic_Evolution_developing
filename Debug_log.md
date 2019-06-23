# 主题演化测试log

## 6.23

- 测试其在修改train文件的效果

- debug部分

- 完成模型测试 上传在google-drive 的topic-tes1 但是貌似丢失了utils的文件（==待检查==）

- explore_pyLDAvis的模型测试上次已经做过

  - 下面这个应该是最后的有关pyLDAvis的最后的模型组件，应该还需要在原来的preproces做相关的处理。

  - ```python
    # 注意这里需要topics文件，这个是lda2vec的原作者文件
    import torch
    import numpy as np
    import topics
    
    def softmax(x):
        # x has shape [batch_size, n_classes]
        e = np.exp(x)
        n = np.sum(e, 1, keepdims=True)
        return e/n
    
    
    state = torch.load('model_state.pytorch', map_location=lambda storage, loc: storage)
    n_topics = 25
    
    doc_weights = state['doc_weights.weight'].cpu().clone().numpy()
    topic_vectors = state['topics.topic_vectors'].cpu().clone().numpy()
    resulted_word_vectors = state['neg.embedding.weight'].cpu().clone().numpy()
    
    # distribution over the topics for each document
    topic_dist = softmax(doc_weights)
    
    # vector representation of the documents
    doc_vecs = np.matmul(topic_dist, topic_vectors)
    
    # 单词字典对应数字
    decoder = np.load('decoder.npy')[()]
    
    data = topics.prepare_topics(doc_weights, topic_vectors, resulted_word_vectors, decoder)
    
    
    
    doc_lengths = np.load("doc_lengths.npy")
    term_frequency = np.load("term_frequency.npy")
    data['doc_lengths'] = doc_lengths
    data['term_frequency'] = term_frequency
    np.savez('topics.pyldavis', **data)
    ```

  - 

  - 



### DEBUG

- ```python
  self.multinomial.draw(batch_size*window_size*self.num_sampled)
  ￼ File "F:\备份\临时备份\Topic_evolution_developing\utils\alias_multinomial.py", line 57, in draw
    b = torch.bernoulli(q)
  
  builtins.RuntimeError: invalid argument 1: must be >= 0 and <= 1 at ..\aten\src\TH\THRandom.cpp:320
  #与此有关的是 index_select
          K = self.J.size(0)
          r = torch.LongTensor(np.random.randint(0, K, size=N))##.cuda()
          q = self.q.index_select(0, r)
          j = self.J.index_select(0, r)
          b = torch.bernoulli(q)
          oq = r.mul(b.long())
          oj = j.mul((1 - b).long())
          
  ```

- 修改两行代码，待更新，也可一直看github上的版本更新

## 6.22
- 工程迁移

- 检测当前的cuda是否安装成功

  - ```
    import torch
    print(torch.cuda.is_available())
    ```

  - RuntimeError: CUDA error: device-side assert triggered

  - ```
    epoch 1
      0%|                                                                                                                                                                                 | 0/83 [00:00<?, ?it/s]/home/g812126839qq/Topic_evolution_developing/utils/lda2vec_loss.py:196: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
      doc_probs = F.softmax(doc_weights)
    THCudaCheck FAIL file=/pytorch/aten/src/THC/THCCachingHostAllocator.cpp line=265 error=59 : device-side assert triggered
    
    Traceback (most recent call last):
      File "train.py", line 36, in <module>
        main()
      File "train.py", line 32, in main
        save_every=20, grad_clip=5.0
      File "/home/g812126839qq/Topic_evolution_developing/utils/training.py", line 127, in train
        neg_loss, dirichlet_loss = model(doc_indices, pivot_words, target_words)
      File "/usr/local/lib/python3.5/dist-packages/torch/nn/modules/module.py", line 493, in __call__
        result = self.forward(*input, **kwargs)
      File "/home/g812126839qq/Topic_evolution_developing/utils/lda2vec_loss.py", line 72, in forward
        neg_loss = self.neg(pivot_words, target_words, doc_vectors, w)
      File "/usr/local/lib/python3.5/dist-packages/torch/nn/modules/module.py", line 493, in __call__
        result = self.forward(*input, **kwargs)
      File "/home/g812126839qq/Topic_evolution_developing/utils/lda2vec_loss.py", line 156, in forward
        sum_log_sampled = (noise*unsqueezed_context).sum(3).neg().sigmoid()\
    RuntimeError: CUDA error: device-side assert triggered
    ```

  - 本地no_cuda调试

  - ```
    ￼ File "E:\work\github\Topic_evolution_developing\utils\alias_multinomial.py", line 65, in draw
      b = torch.bernoulli(q)
    
    builtins.RuntimeError: invalid argument 1: must be >= 0 and <= 1 at ..\aten\src\TH\THRandom.cpp:320
    ```

    - 这个在上次做中文语料的时候应该也是有debug类似的东西，现在我貌似找到了上次的东西应该可以进一步做相关的 工作了，不知道能不能兼容，在google-drive的new1.zip

- 