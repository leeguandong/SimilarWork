from .module import Module


class Sequential(Module):
    def __init__(self, *sequences):
        super(Sequential, self).__init__()

        self.module_list = list(sequences)
        for i_mod, module in enumerate(self.module_list):
            try:
                module_state_dict = module.get_state_dict()
                updated_names = [
                    (f"sequential.{i_mod}.{k}", v) for k, v in module_state_dict.items()]
                self.register_parameter(*updated_names)
            except AttributeError:
               pass

    def forward(self, *input):
        """
        分两步写，*拆包多个参数
        :param input:
        :return:
        """
        out = self.module_list[0](*input)
        for module in self.module_list[1:]:
            out = module(out)
        return out
