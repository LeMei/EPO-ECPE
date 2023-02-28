# EPO-ECPE

**Emotion Prediction Oriented method with Multiple Supervisions for Emotion-Cause Pair Extraction**

**accepted to Transactions on Audio, Speech and Language Processing**

**Abstract**

Emotion-cause pair extraction (ECPE) task aims to extract all the pairs of emotions and their causes from an unannotated emotion text. The previous works usually extract the emotion-cause pairs from two perspectives of emotion and cause. However, emotion extraction is more crucial to the ECPE task than cause extraction. Motivated by this analysis, we propose an end-to-end emotion-cause extraction approach oriented toward emotion prediction (EPO-ECPE), aiming to fully exploit the potential of emotion prediction to enhance emotion-cause pair extraction. Considering the strong dependence between emotion prediction and emotion-cause pair extraction, we propose a synchronization mechanism to share their improvement in the training process. That is, the improvement of emotion prediction can facilitate the emotion-cause pair extraction, and then the results of emotion-cause pair extraction can also be used to improve the accuracy of emotion prediction simultaneously. For the emotion-cause pair extraction, we divide it into genuine pair supervision and fake pair supervision, where the genuine pair supervision learns from the pairs with more possibility to be emotion-cause pairs. In contrast, fake pair supervision learns from other pairs. In this way, the emotion-cause pairs can be extracted directly from the genuine pair, thereby reducing the difficulty of extraction. Experimental results show that our approach outperforms the 13 compared systems and achieves new state-of-the-art performance.

arxiv: https://arxiv.org/pdf/2302.12417.pdf
