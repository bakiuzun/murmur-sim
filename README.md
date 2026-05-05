# murmur-sim V1 Genesis
Minimal version of a drone waypoints following task with Genesis. 

## murmur-sim V2 
Currently, I am working on integrating camera rendering for an autonomous UAV using human prompts. The first task is to have the agent search for a particular object. The agent should understand which features to look for to determine a 'direction'—for example, if the object is typically located near the trees, it should search around the trees and not the whole city. If successful the UAV will then be  generalized to different tasks.



### First test 
Taking different angles of a target point -> retrieve dino features and save as memory features. During training do a cos sim with features from camera and the bank. More similarity -> higher reward. The drone should understand that it should keep itself around the target. 


### Second test
Same as first test but without using any feature extractor. I will feed raw pixels and the agent will receive reward if the object is visible in it's camera (I will use segmentation proposed by genesis)


### Third test 
Inlcude some way for the agent to "understand" where he navigated. Some penalty will be given he if passes multiple times to the same place but of course he should understand the why he is getting penalty.


