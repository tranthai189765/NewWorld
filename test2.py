import mate 

env = mate.make('MultiAgentTracking-v0')
env = mate.MultiCamera(env, target_agent=mate.IPPOTargetAgent)