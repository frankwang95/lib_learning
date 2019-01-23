import types


"""
Attributes
    n_latent_layer
    latent_layer
New Methods
    encode
    decode
    * Note post process attribute does nothing in the encode and decode functions.
Changes
    rebase
"""


def convert_to_autoencoder(nn, n_latent_layer):
    nn.n_latent_layer = n_latent_layer


    def encode(self, in_vector):
        return self.session.run(self.latent_layer, feed_dict={self.input: in_vector})


    def decode(self, in_vector):
        return self.session.run(self.output, feed_dict={self.latent_layer: in_vector})


    def rebase(self, input_vector=None, train_targets_vector=None, latent_layer=None):
        # == Input/LatentLayer/Output Handles == #
        if input_vector is not None:
            self.input = input_vector
            self.latent_layer = self.feed_forwards(self.input, end_layer=self.n_latent_layer)
            self.output = self.feed_forwards(self.latent_layer, start_layer=self.n_latent_layer)

        # == Training == #
        if train_targets_vector is not None:
            self.train_targets = train_targets_vector
            self.create_train_step()

        # == Monitors == #
        self.monitors = [m.link_to_network(self) for m in self.monitors]


    nn.encode = types.MethodType(encode, nn)
    nn.decode = types.MethodType(decode, nn)
    nn.rebase = types.MethodType(rebase, nn)
    nn.rebase(nn.input)

    return nn
