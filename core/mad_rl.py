class MAD_RL:
    @staticmethod
    def engine(engine_parameters=None, agent_parameters=None,
               shared_agent=None):
        module = engine_parameters["module"]
        class_name = engine_parameters["class"]

        environment_module = __import__(module, fromlist=["*"])
        environment_class = getattr(environment_module, class_name)

        return environment_class(
            engine_parameters, agent_parameters, shared_agent)

    @staticmethod
    def agent(parameters=None):
        module = parameters["module"]
        class_name = parameters["class"]

        agent_mod = __import__(module, fromlist=["*"])
        agent_class = getattr(agent_mod, class_name)

        return agent_class(parameters)
