import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel_MTP(nn.Module):
    """TSP Model with Multi-Token Prediction (MTP) capability"""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        
        # MTP specific parameters
        self.mtp_depth = model_params.get('mtp_depth', 2)  # Number of future tokens to predict
        self.mtp_weight = model_params.get('mtp_weight', 0.5)  # Weight for MTP losses
        
        # Main model components (shared)
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        
        # MTP modules - one for each prediction depth
        self.mtp_modules = nn.ModuleList([
            MTP_Module(depth=k, **model_params) for k in range(1, self.mtp_depth + 1)
        ])
        
        # Shared output head for all predictions
        embedding_dim = model_params['embedding_dim']
        self.shared_output_head = nn.Linear(embedding_dim, 1, bias=True)
        
        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step, repair=False):
        """
        Forward pass with MTP capability
        Returns: selected_teacher, prob, mtp_losses, selected_student
        """
        batch_size_V = state.data.size(0)
        problem_size = state.data.shape[1]
        
        if self.mode == 'train':
            # Standard next-token prediction
            encoded_nodes = self.encoder(state.data)
            probs = self.decoder(encoded_nodes, selected_node_list)
            
            selected_student = probs.argmax(dim=1)
            selected_teacher = solution[:, current_step - 1]
            prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)
            
            # MTP predictions and losses
            mtp_losses = self._compute_mtp_losses(
                encoded_nodes, selected_node_list, solution, current_step, problem_size
            )
            
            return selected_teacher, prob, mtp_losses, selected_student
            
        elif self.mode == 'test':
            if not repair:
                if current_step <= 1:
                    self.encoded_nodes = self.encoder(state.data)
                probs = self.decoder(self.encoded_nodes, selected_node_list)
            else:
                if current_step <= 2:
                    self.encoded_nodes = self.encoder(state.data)
                probs = self.decoder(self.encoded_nodes, selected_node_list)
            
            selected_student = probs.argmax(dim=1)
            selected_teacher = selected_student
            prob = 1
            mtp_losses = {}  # No MTP losses in test mode
            
            return selected_teacher, prob, mtp_losses, selected_student

    def _compute_mtp_losses(self, encoded_nodes, selected_node_list, solution, current_step, problem_size):
        """Compute MTP losses for multiple future prediction depths"""
        mtp_losses = {}
        batch_size = encoded_nodes.shape[0]
        
        # Current representation from main model
        current_repr = self._get_main_model_representation(encoded_nodes, selected_node_list)
        
        prev_repr = current_repr
        
        for depth in range(1, self.mtp_depth + 1):
            # Check if we can predict this far ahead
            target_step = current_step - 1 + depth
            if target_step >= problem_size:
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                continue
            
            target_cities = solution[:, target_step]  # [B]
            
            # Validate target cities are within bounds
            if torch.any(target_cities >= problem_size) or torch.any(target_cities < 0):
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                continue
            
            current_cities= solution[:, target_step - 1] if target_step > 0 else solution[:, 0]
            
            embeddings = self._get_city_embeddings(encoded_nodes, current_cities)  # [B, emb_dim]
            
            # Pass through MTP module
            mtp_repr = self.mtp_modules[depth-1](prev_repr, embeddings, selected_node_list, depth)
            
            # Generate probabilities using shared output head
            mtp_probs = self._generate_mtp_probabilities(
                mtp_repr, encoded_nodes, selected_node_list, depth, current_step
            )
            
            # Check for valid logits (avoid all -inf situation)
            if torch.all(torch.isinf(mtp_probs)):
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                continue
            # Compute cross-entropy loss for this depth
            try:
                mtp_loss = F.cross_entropy(
                    mtp_probs, target_cities.long(), reduction='mean'
                )
                # Check if loss is valid
                if torch.isnan(mtp_loss) or torch.isinf(mtp_loss):
                    mtp_loss = torch.tensor(0.0, device=encoded_nodes.device)
                mtp_losses[f'mtp_depth_{depth}'] = mtp_loss
            except Exception as e:
                # Fallback if cross-entropy fails
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
            mtp_losses[f'mtp_depth_{depth}'] = mtp_loss
            
            # Update representation for next depth
            prev_repr = mtp_repr
            
        return mtp_losses

    def _get_main_model_representation(self, encoded_nodes, selected_node_list):
        """Get representation from main model for MTP input"""
        # Use the decoder's internal representation
        # This is a simplified version - in practice, you might want to extract 
        # intermediate representations from the decoder
        batch_size = encoded_nodes.shape[0]
        embedding_dim = encoded_nodes.shape[2]
        
        if selected_node_list.shape[1] == 0:
            # No cities selected yet, use mean of all nodes
            return encoded_nodes[:, 0, :]  # [B, emb_dim]
        else:
            # Use representation of last selected city
            last_selected = selected_node_list[:, -1]  # [B]
            gathering_index = last_selected[:, None, None].expand(batch_size, 1, embedding_dim)
            last_repr = encoded_nodes.gather(dim=1, index=gathering_index).squeeze(1)  # [B, emb_dim]
            return last_repr

    def _get_city_embeddings(self, encoded_nodes, city_indices):
        """Get embeddings for specific cities"""
        batch_size = encoded_nodes.shape[0]
        embedding_dim = encoded_nodes.shape[2]
        
        gathering_index = city_indices[:, None, None].expand(batch_size, 1, embedding_dim)
        city_embeddings = encoded_nodes.gather(dim=1, index=gathering_index).squeeze(1)  # [B, emb_dim]
        return city_embeddings

    def _generate_mtp_probabilities(self, mtp_repr, encoded_nodes, selected_node_list, depth, current_step):
        """Generate probability distribution over cities for MTP prediction"""
        batch_size = mtp_repr.shape[0]
        problem_size = encoded_nodes.shape[1]
        
        # Get the appropriate MTP module for this depth
        mtp_module = self.mtp_modules[depth - 1]
        
        # Transform query and keys using depth-specific linear layers
        # mtp_repr: [B, emb_dim], encoded_nodes: [B, V, emb_dim]
        transformed_query = mtp_module.query_transform(mtp_repr)  # [B, emb_dim]
        transformed_keys = mtp_module.key_transform(encoded_nodes)  # [B, V, emb_dim]
        
        # Compute logits for all cities using transformed representations
        logits = torch.bmm(transformed_query.unsqueeze(1), transformed_keys.transpose(1, 2)).squeeze(1)  # [B, V]

        # Create mask for visited cities and apply it
        if selected_node_list.shape[1] > 0:
            # Create mask for already selected cities (should be -inf)
            selected_mask = torch.zeros(batch_size, problem_size, device=logits.device, dtype=torch.bool)
            
            # Handle potential out-of-bounds indices
            valid_indices = (selected_node_list >= 0) & (selected_node_list < problem_size)
            if torch.any(valid_indices):
                valid_selected = selected_node_list[valid_indices]
                batch_indices = torch.arange(batch_size, device=logits.device)[:, None].expand_as(selected_node_list)[valid_indices]
                selected_mask[batch_indices, valid_selected] = True
            
            # Apply masking: set selected cities to -inf
            logits = logits.masked_fill(selected_mask, float('-inf'))
            
        logits = F.softmax(logits, dim=1)  # Convert logits to probabilities
        
        # Ensure at least one city is selectable (prevent all -inf)
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        if torch.any(torch.isinf(max_logits)):
            # If all logits are -inf, reset to zeros for this batch
            inf_batch_mask = torch.isinf(max_logits).squeeze(1)
            logits[inf_batch_mask] = 0.0
        
        return logits


class MTP_Module(nn.Module):
    """Multi-Token Prediction Module for predicting future cities"""
    
    def __init__(self, depth, **model_params):
        super().__init__()
        self.depth = depth
        embedding_dim = model_params['embedding_dim']
        
        # Input projection - combines previous representation and target embedding
        self.input_projection = nn.Linear(embedding_dim * 2, embedding_dim, bias=True)
        self.input_norm = nn.LayerNorm(embedding_dim)
        
        # Transformer block for this depth
        self.transformer_block = MTP_TransformerBlock(**model_params)
        
        # Linear layers for bmm transformation
        self.query_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
    def forward(self, prev_repr, target_embedding, selected_node_list, depth):
        """
        Args:
            prev_repr: [B, emb_dim] - representation from previous depth
            target_embedding: [B, emb_dim] - embedding of city to predict
            selected_node_list: [B, current_step] - already selected cities
            depth: int - prediction depth
        Returns:
            mtp_repr: [B, emb_dim] - representation for this MTP depth
        """
        # Combine inputs
        combined_input = torch.cat([
            self.input_norm(prev_repr), 
            self.input_norm(target_embedding)
        ], dim=-1)  # [B, 2*emb_dim]
        
        # Project to embedding dimension
        projected_input = self.input_projection(combined_input)  # [B, emb_dim]
        
        # Pass through transformer block
        mtp_repr = self.transformer_block(projected_input.unsqueeze(1)).squeeze(1)  # [B, emb_dim]
        
        return mtp_repr


class MTP_TransformerBlock(nn.Module):
    """Transformer block used in MTP modules"""
    
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=head_num,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, emb_dim]
        Returns:
            output: [B, seq_len, emb_dim]
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


# Import the original classes from TSPModel
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = 1
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])

        self.k_1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):
        list = selected_node_list
        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)
        new_list_len = prob_size - list.shape[1]
        index_2 = list.type(torch.long)
        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])
        new_list[index_1, index_2] = -2
        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        new_data = data
        emb_dim = data.shape[-1]
        new_data_len = new_list_len
        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)
        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])
        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))
        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_

    def _get_encoding(self, encoded_nodes, node_index_to_pick):
        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)
        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        return picked_nodes

    def forward(self, data, selected_node_list):
        batch_size_V = data.shape[0]
        problem_size = data.shape[1]
        new_data = data

        left_encoded_node = self._get_new_data(new_data, selected_node_list, problem_size, batch_size_V)

        first_and_last_node = self._get_encoding(new_data, selected_node_list[:, [0, -1]])
        embedded_first_node_ = first_and_last_node[:, 0]
        embedded_last_node_ = first_and_last_node[:, 1]

        embedded_first_node_ = self.embedding_first_node(embedded_first_node_)
        embedded_last_node_ = self.embedding_last_node(embedded_last_node_)

        out = torch.cat((embedded_first_node_.unsqueeze(1), left_encoded_node, embedded_last_node_.unsqueeze(1)), dim=1)

        for layer in self.layers:
            out = layer(out)

        out = self.Linear_final(out).squeeze(-1)
        out[:, [0, -1]] = out[:, [0, -1]] + float('-inf')

        props = F.softmax(out, dim=-1)
        props = props[:, 1:-1]

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)
        props = props_clone

        new_props = torch.zeros(batch_size_V, problem_size)
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, selected_node_list.shape[1])
        index_2_ = selected_node_list.type(torch.long)
        new_props[index_1_, index_2_] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        new_props[index] = props.ravel()

        return new_props


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))