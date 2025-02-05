from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, Model

class CausalConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        
        """
        Initializes the CausalConv1D layer.
        
        Args:
            filters (int): The number of filters (kernels) to use in the Conv1D layer.
            kernel_size (int): The size of the convolution kernel.
            dilation_rate (int, optional): The dilation rate for dilated convolutions, which expands the kernel to cover 
                                           a wider range of input elements. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        
        super(CausalConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        # Create a Conv1D layer with causal padding.
        # Causal padding ensures that the output at time 't' only depends on inputs at or before time 't',
        # which is crucial for time-series or sequence data to prevent future information leakage.
        self.conv = layers.Conv1D(filters=filters,  kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)
    
    def call(self, inputs):

        """
        Applies the causal convolution to the input tensor.
        
        Args:
            inputs (Tensor): The input tensor.
            
        Returns:
            Tensor: The output tensor after applying the causal convolution.
        """
        
        return self.conv(inputs)


class ConditionalWaveNetBlock(layers.Layer):

    """
    Initializes a conditional WaveNet block.

    Args:
        filters (int): The number of filters for the convolutional layers.
        kernel_size (int): The size of the convolution kernel.
        dilation_rate (int, optional): The dilation rate for the dilated convolution. Default is 1.
        **kwargs: Additional keyword arguments passed to the parent class.
    """
    
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ConditionalWaveNetBlock, self).__init__(**kwargs)

         # Create a dilated causal convolution block.
        self.dilated_conv = CausalConv1D(filters, kernel_size, dilation_rate)
        self.condition_proj = layers.Dense(2 * filters)

        # 1x1 convolution layers for generating skip and residual connections.
        self.skip_connection = layers.Conv1D(filters, 1)
        self.residual_connection = layers.Conv1D(filters, 1)

        # A transposed convolution layer for learnable upsampling of the conditional input.
        self.upsample = layers.Conv1DTranspose(2 * filters, kernel_size=8, strides=4, padding='same')
        self.filters = filters
        
    def call(self, inputs, training=False):
        """
        Executes the forward pass of the conditional WaveNet block.

        Args:
            inputs (tuple): A tuple (x, cond) where:
                - x: The main input tensor.
                - cond: The conditional tensor (can be None if no conditioning is used).
            training (bool, optional): If True, random noise is added during the dilated convolution for training robustness.

        Returns:
            tuple: A tuple (residual_output, skip_output) where:
                - residual_output: The output for the residual connection.
                - skip_output: The output for the skip connection.
        """
        
        x, cond = inputs  # Unpack as tuple to ensure conditioning is passed properly
        
        # Apply the dilated causal convolution.
        x = self.dilated_conv(x)
                
        # Process the conditional input if it is provided.
        if cond is not None:
            cond_shape = tf.shape(cond)

            # Handle (batch, con_dim) shape
            if len(cond_shape) == 2:  
                cond = self.condition_proj(cond)
                cond = tf.expand_dims(cond, axis=1)
                # Tile the conditional input along the time axis to match the main input x.
                cond = tf.tile(cond, [1, tf.shape(x)[1], 1])

            # Handle (batch, time//hop, con_dim) shape
            elif len(tf.shape(cond)) == 3:
                cond_flat = tf.reshape(cond, [-1, cond_shape[-1]]) # Reshape to (batch * time//hop, con_dim)
                cond_proj = self.condition_proj(cond_flat)
                cond = tf.reshape(cond_proj, [cond_shape[0], cond_shape[1], 2 * self.filters]) # Reshape back to (batch, time//hop, 2*filters)
                cond = self.upsample(cond) # Upsample the conditional input to match the time resolution of x.
                
                # Adjust the time dimension of the conditional input to exactly match that of x.
                target_length = tf.shape(x)[1]
                current_length = tf.shape(cond)[1]

                # If the conditional input is longer than needed, trim it.
                if current_length > target_length:
                    cond = cond[:, :target_length, :]
                # If it is shorter, pad it with zeros.
                elif current_length < target_length:
                    padding = [[0, 0], [0, target_length - current_length], [0, 0]]
                    cond = tf.pad(cond, padding)
    
            cond_tanh, cond_sigmoid = tf.split(cond, 2, axis=-1)

            # Combine the main input x with the conditional projections using element-wise addition.
            tanh = tf.tanh(x + cond_tanh)
            sigmoid = tf.sigmoid(x + cond_sigmoid)
        else:
            # If no conditional input is provided, use x directly for both gating mechanisms.
            tanh = tf.tanh(x)
            sigmoid = tf.sigmoid(x)

        # Element-wise multiplication of the tanh and sigmoid outputs to form the gated activation.
        gates = tanh * sigmoid
        
        skip = self.skip_connection(gates)
        residual = self.residual_connection(gates)
        
        return residual + x, skip # Return the combined residual output (adding the original input x) and the skip output.


class ConditionalWaveNet(Model):
    
    def __init__(self, num_blocks=10, filters=32, kernel_size=2, condition_dim=None, num_classes=None):
        """
        Initializes the ConditionalWaveNet model.
    
        Args:
            num_blocks (int): The number of WaveNet blocks to stack.
            filters (int): The number of filters used in the convolutional layers.
            kernel_size (int): The size of the convolution kernels.
            condition_dim (int, optional): The dimensionality of the conditional input.
            num_classes (int, optional): The number of classes for discrete conditioning. 
            If provided, an embedding layer will be used.
        """
        
        super(ConditionalWaveNet, self).__init__()
        
        self.num_blocks = num_blocks
        self.filters = filters
        self.condition_dim = condition_dim
        
        # Initial convolution
        self.init_conv = layers.Conv1D(filters, 1)
        
        self.embedding = None
        # If discrete conditioning is provided, set up an Embedding layer.
        if num_classes is not None:
            self.embedding = layers.Embedding(num_classes, condition_dim)
        
        # If a continuous condition is used, create a Dense layer to project the condition.
        if condition_dim is not None:
            self.condition_proj = layers.Dense(condition_dim)
        
        # Create a list of WaveNet blocks, each with an increasing dilation rate.
        # The dilation rate for each block is set as 2**i.
        self.blocks = [ConditionalWaveNetBlock(filters, kernel_size, 2**i) for i in range(num_blocks)]
        
        # Define the output layers:
        # Two intermediate 1x1 convolutions with ReLU activation,
        # followed by a final 1x1 convolution that outputs 256 channels (raw logits).
        self.final_conv1 = layers.Conv1D(filters, 1, activation='relu')
        self.final_conv2 = layers.Conv1D(filters, 1, activation='relu')
        self.output_conv = layers.Conv1D(256, 1)  # Remove softmax activation
        
    def process_condition(self, condition):
        """
        Processes the conditional input.

        Depending on whether a discrete or continuous condition is provided,
        it passes the condition through an embedding layer or a dense layer.

        Args:
            condition (Tensor or None): The conditional input.

        Returns:
            Tensor or None: The processed conditional input, or None if no condition is provided.
        """        
        if condition is None:
            return None

        # If an embedding layer exists (discrete conditions), use it.
        if self.embedding is not None:
            condition = self.embedding(condition)
        else:
            # Otherwise, project the continuous condition using a dense layer.
            condition = self.condition_proj(condition)
            
        return condition
        
    def call(self, inputs, training=True):
        
        """
        Performs the forward pass of the ConditionalWaveNet model.

        Args:
            inputs (Tensor or tuple): Either a single input tensor 'x' or a tuple (x, condition),
                                      where 'condition' is the additional conditional input.

        Returns:
            Tensor: The output logits from the model.
        """

        # Check if inputs contain a condition. Unpack accordingly.
        if isinstance(inputs, (list, tuple)):
            x, condition = inputs
        else:
            x, condition = inputs, None

        # Process the conditional input if provided.
        condition = self.process_condition(condition)
        
        x = self.init_conv(x)

        # Create a list to collect skip connections from each WaveNet block.
        skip_connections = []
        for block in self.blocks:
            x, skip = block([x, condition])  # Pass as tuple
            skip_connections.append(skip)

        # Sum all skip connections; note that this includes the skip from the final block.
        # This aggregated skip output will be fed into the post-processing layers.
        out = tf.add_n(skip_connections)

        # Pass the aggregated skip connections through the final post-processing convolution layers.
        out = self.final_conv1(out)
        out = self.final_conv2(out)
        out = self.output_conv(out)  # Raw logits

        if not training:
            return tf.argmax(out, axis=-1)
        
        return out
        
    def predict(self, inputs, batch_size=32,  steps=None, callbacks=None):
        results = []
        num_batches = (len(inputs) + batch_size - 1) // batch_size
          
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            results.append(self.call(batch, training=False))
        
        return tf.concat(results, axis=0)

        

def create_and_train_model(x_train, y_train, x_val, y_val, conditions_train, conditions_val, condition_dim=80, num_classes=None, epochs=2000, batch_size=1000):
    """
    Creates and trains a conditional WaveNet model with proper loss handling.

    Args:
        x_train (Tensor): Training input data.
        y_train (Tensor): Training target data.
        x_val (Tensor): Validation input data.
        y_val (Tensor): Validation target data.
        conditions_train (Tensor): Conditional training input data.
        conditions_val (Tensor): Conditional validation input data.
        condition_dim (int, optional): Dimensionality of the condition projection. Default is 80.
        num_classes (int, optional): Number of classes for discrete conditions. If provided, an embedding layer is used.
        epochs (int, optional): Number of epochs for training. Default is 2000.
        batch_size (int, optional): Batch size for training. Default is 1000.

    Returns:
        model (ConditionalWaveNet): The trained conditional WaveNet model.
        history (History): A record of training loss and metrics.
    """
    
    # Create model
    model = ConditionalWaveNet(condition_dim=condition_dim,  num_classes=num_classes)
    
    # Compile with sparse categorical crossentropy and proper from_logits setting
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    # Checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./wavenet_best.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    
    # Train model
    history = model.fit([x_train, conditions_train], y_train, 
                        validation_data=([x_val, conditions_val], y_val), 
                        epochs=epochs, batch_size=batch_size, callbacks=[checkpoint_callback])
    
    return model, history
    