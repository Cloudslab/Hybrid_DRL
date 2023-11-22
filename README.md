# Hybrid_DRL

The implementation of an adaptation of the actor critic algorithm A2C with Proximal Policy Optimization (PPO) for policy optimization. Here we use a hierarchical action space with a multi-actor network capable of decision making at two levels. This work is aimed at training an optimal request scheduling policy in a serverless and serverful hybrid environment to minimize execution cost and time for users. The implementation is done on an event-driven simulator developed in python using the OpenAI gym environment and Keras for DRL model training.
