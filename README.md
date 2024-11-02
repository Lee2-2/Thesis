# Mutual Learning Approaches with Federated Learning to Detect Low-rate and High-rate DDoS Attacks in IoT Networks

本論文提出了一種個性化聯邦學習（Personalized Federated Learning, PFL）方法，用於偵測高頻與低頻 DDoS 攻擊。此 PFL 框架透過在客戶端進行個性化模型訓練，有效降低因資料分布不均而產生的性能差異，提升模型的準確性與穩定性。此外，優化的相互學習機制允許神經網路自動減少參與模型數量，以降低通訊成本和訓練時間，同時提升準確率。

# Requirements
* torch==1.13.1
* torchvision==0.14.1
* pandas==1.4.4
* numpy==1.22.4

# Dataset
使用CICDDOS2019、CICIDS2017分別當作高頻率與低頻率的DDoS攻擊的資料集。

# Code
此程式依據聯邦學習架構進行設計，主要執行流程由以下函式組成：

create_worker()：建立客戶端並分配初始模型。

clustering()：使用 DBSCAN 分群演算法將客戶端進行分群，以便在資料分布不均的情況下改善模型性能。

decide_other_model()：選擇與本地端同一群的模型，以利後續相互學習。

send_model()：將選定的模型發送至其他server端。

local_training()：客戶端的本地模型與server端的模型相互學習以提升模型準確性。

aggregate()：聚合更新後的模型以提高整體準確性。

return_model()：將最終模型返回給客戶端。


其中，分群使用 DBSCAN 演算法，並在本地訓練（local training）中結合相互學習（mutual learning）機制，以提升模型的穩定性和泛化能力。



