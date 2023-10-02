# SplitFed: When Federated Learning Meets Split Learning

---
### Abstract
**Proposal:** combine Split learning and Federated Learning to eliminate their drawbacks

**Why:**

- SL provides better model privacy than FL due to the machine learning model architecture split between clients and the server.
- The split model makes SL a better option for resource-constrained environments.
- SL performs slower than FL due to the relay-based training across multiple clients.


In addition, the architecture is supposed to incorporate differential privacy and PixelDP to enhance data privacy and model robustness.

**Methods:**
FL, SL, and SFL are evaluated based on comparative performance measurements on four standard datasets and four popular
models

**Results:**
- SFL offers better model privacy than FL
- (pure) SFL provides similar test accuracy and communication efficiency as SL
- SFL decreases its computation time per global epoch over SL for multiple clients.
- as in SL, the communication efficiency of SFL over FL improves with the number of clients.

---
### Introduction

- The main advantage of FL is that it allows parallel, hence efficient, ML model training across many clients.
- The main disadvantage of FL is that each client needs to run the full ML model, which may be an issue with resource-constrained clients
- There is a privacy concern from the model's privacy perspective during training because the server and clients have full access to the local and global models.

---

- SL: Assigning only a part of the network to train at the client-side reduces processing load (compared to that of running a complete network as in FL)
- The synchronization of the learning process with multiple clients is done either in a centralized mode or peer-to-peer mode in SL
- A client has no access to the server-side model and vice-versa
- relay-based training in SL makes the clients' resources idle because only one client engages with the server at one instance;
causing a significant increase in the training overhead with many clients.

---


### Proposed Framweork

- SFL combines the primary strength of FL, which is parallel processing among distributed clients, and the primary strength of SL, which is network splitting into client-side and server-side sub-networks during training
- use SFL in resource-constrained environments
- use SFL if fast model training time is required to periodically update the global model based on a continually updating dataset over time (e.g., data stream)
- Unlike SL, all clients carry out their computations in parallel and engage with the main server and fed server
- The fed server is introduced to conduct FedAvg on the client-side local updates
- the fed server synchronizes the client-side global model in each round of network training


**Workflow:**
1. All clients perform forward propagation on their client-side model in parallel, including its noise layer, and pass their smashed data to the main server.
2. Then the main server processes the forward propagation and back-propagation on its server-side model with each client's smashed data separately in (somewhat) parallel.
3. It then sends the gradients of the smashed data to the respective clients for their back-propagation.
4. Afterward, the server updates its model by FedAvg, i.e., weighted averaging of gradients that it computes during the back-propagation on each client's smashed data.
5. At the client's side, after receiving the gradients of its smashed data, each client performs the back-propagation on their client-side local model and computes its gradients.
6. A DP mechanism is used to make these gradients private and send them to the fed server.
7. The fed server conducts the FedAvg of the client-side local updates and sends them back to all participating clients.



**Privacy**
- A network split in ML learning enables the clients/fed server and the main server to maintain the full model privacy by not allowing the main server to get the client-side model updates and vice-versa
- The possibility of inferring the client-side model parameters and raw data is highly unlikely if we configure the client-side ML networks' fully connected layers *with sufficiently large numbers of nodes* .
- However, for a smaller client-side network, the possibility of this issue can be high. This issue can be controlled by modifying the loss function at the client-side
- if any server/client becomes curious: apply two measures (i) differential privacy to the client-side model training and (ii) PixelDP noise layer in the client-side model.
- For Privacy Protection on Fed Server: add calibrated noise to the average gradient
- Privacy Protection on Main Server: integrate a noise layer in the client-side model based on the concepts of PixelDP

### Experiments

- analyze the total communication cost and model training time for FL, SL, and SFL under a uniform data distribution
- Datasets: HAM10000, MNIST, FMIST, CIFAR10 -> all images of at least 28x28
- Architectures: LeNet, AlexNet, VGG16, ResNet18
