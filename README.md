# murmur-sim V1 Genesis
Minimal version of a drone waypoints following task with Genesis. 

## murmur-sim V2 
Currently, I am working on integrating camera rendering for an autonomous UAV using human prompts. The first task is to have the agent search for a particular object. The agent should understand which features to look for to determine a 'direction'—for example, if the object is typically located near the trees, it should search around the trees and not the whole city. If successful the UAV will then be  generalized to different tasks.


### First test
Capture the target point from different angles, extract DINO features, and store them as a memory bank. During training, compute the cosine similarity between the camera features and the bank: higher similarity yields higher reward. The drone is expected to learn to keep itself oriented toward the target.

### Second test
Same setup as the first test, but without any feature extractor. The agent is fed raw pixels and receives a reward when the object is visible in its camera (visibility is determined using the segmentation provided by Genesis).

### Conclusion
Both tests failed: the drone was unable to perform the task. The likely causes are twofold. First, the reward definition: in the first test, cosine similarity may be too noisy or weakly discriminative to provide a usable gradient, and in the second test, the visibility-based reward is sparse and binary, so the agent rarely (if ever) receives a signal early on and has nothing to learn from. Second, feeding raw DINOv2 features to the agent is likely too hard: the agent has to learn both to interpret a high-dimensional embedding it did not learn itself and to derive a control policy, all from a weak reward.
Using an object detector and feeding the agent target coordinates instead of raw features would most likely have succeeded. However, that is not the goal here: the aim is to feed the agent an embedding and have it understand, from that embedding alone, what to do. This remains an open and difficult direction; a more realistic path may be to start from a simpler, structured input and progressively enrich it toward a learned semantic representation.


### Next step 
Implementing DreamerV3 for the Drone
