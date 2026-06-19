import torch
import torch.nn as nn
import torch.nn.functional as F


class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']

        # MTP specific parameters
        self.mtp_depth = model_params.get('mtp_depth', 2)
        self.mtp_weight = model_params.get('mtp_weight', 0.5)

        # Main model components
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        # MTP modules
        self.mtp_modules = nn.ModuleList([
            MTP_Module(depth=k, **model_params) for k in range(1, self.mtp_depth + 1)
        ])
        
        # Store the decoder output for MTP
        self.decoder_out = None

        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step, raw_data_capacity=None):
        self.capacity = raw_data_capacity.ravel()[0].item()
        batch_size = state.problems.shape[0]
        problem_size = state.problems.shape[1]
        split_line = problem_size - 1

        def probs_to_selected_nodes(probs_, split_line_, batch_size_):
            selected_node_student_ = probs_.argmax(dim=1)
            is_via_depot_student_ = selected_node_student_ >= split_line_
            not_via_depot_student_ = selected_node_student_ < split_line_

            selected_flag_student_ = torch.zeros(batch_size_, dtype=torch.int)
            selected_flag_student_[is_via_depot_student_] = 1
            selected_node_student_[is_via_depot_student_] = selected_node_student_[
                                                                is_via_depot_student_] - split_line_ + 1
            selected_flag_student_[not_via_depot_student_] = 0
            selected_node_student_[not_via_depot_student_] = selected_node_student_[not_via_depot_student_] + 1
            return selected_node_student_, selected_flag_student_

        if self.mode == 'train':
            remaining_capacity = state.problems[:, 1, 3]
            encoded_nodes = self.encoder(state.problems, self.capacity)
            self.encoded_nodes = encoded_nodes
            probs = self.decoder(encoded_nodes, selected_node_list, self.capacity, remaining_capacity, return_decoder_out=False)

            selected_node_student, selected_flag_student = probs_to_selected_nodes(probs, split_line, batch_size)

            selected_node_teacher = solution[:, current_step, 0]
            selected_flag_teacher = solution[:, current_step, 1]

            is_via_depot = selected_flag_teacher == 1
            selected_node_teacher_copy = selected_node_teacher - 1
            selected_node_teacher_copy[is_via_depot] += split_line

            prob_select_node = probs[torch.arange(batch_size)[:, None], selected_node_teacher_copy[:, None]].reshape(
                batch_size, 1)

            loss_node = -prob_select_node.type(torch.float64).log().mean()

            # MTP predictions and losses using decoder output
            if current_step > 0:
                mtp_losses = self._compute_mtp_losses(
                    selected_node_list, solution, current_step, problem_size, remaining_capacity,
                    state.problems
                )
            else:
                mtp_losses = {}

        if self.mode == 'test':
            probs, split_line = self.get_action_probs(
                state, selected_node_list, raw_data_capacity, current_step
            )

            selected_node_student = probs.argmax(dim=1)
            is_via_depot_student = selected_node_student >= split_line
            not_via_depot_student = selected_node_student < split_line

            selected_flag_student = torch.zeros(batch_size, dtype=torch.int)
            selected_flag_student[is_via_depot_student] = 1
            selected_node_student[is_via_depot_student] = selected_node_student[is_via_depot_student] - split_line + 1
            selected_flag_student[not_via_depot_student] = 0
            selected_node_student[not_via_depot_student] = selected_node_student[not_via_depot_student] + 1

            selected_node_teacher = selected_node_student
            selected_flag_teacher = selected_flag_student
            loss_node = torch.tensor(0, device=state.problems.device)
            mtp_losses = {}

        if self.mode == 'train':
            return loss_node, selected_node_teacher, selected_node_student, selected_flag_teacher, selected_flag_student, mtp_losses
        else:
            return loss_node, selected_node_teacher, selected_node_student, selected_flag_teacher, selected_flag_student

    def get_action_probs(self, state, selected_node_list, raw_data_capacity, current_step):
        if raw_data_capacity is not None:
            self.capacity = raw_data_capacity.ravel()[0].item()
        batch_size = state.problems.shape[0]
        problem_size = state.problems.shape[1]
        split_line = problem_size - 1

        remaining_capacity = state.problems[:, 1, 3]
        if current_step <= 1:
            self.encoded_nodes = self.encoder(state.problems, self.capacity)

        probs = self.decoder(self.encoded_nodes, selected_node_list, self.capacity, remaining_capacity)
        return probs, split_line

    def _compute_mtp_losses(self, selected_node_list, solution, current_step, problem_size,
                            remaining_capacity, problems=None):
        """Compute MTP losses for multiple future prediction depths using decoder output"""
        mtp_losses = {}
        
        # Get encoded_nodes from the main model (stored during forward pass)
        encoded_nodes = self.encoded_nodes
        batch_size = encoded_nodes.shape[0]

        # Get base node representation from decoder output
        base_node_repr = self._get_main_model_representation(encoded_nodes, selected_node_list)
        
        prev_repr = None

        for depth in range(1, self.mtp_depth + 1):
            target_step = current_step + depth
            if target_step >= solution.shape[1]:
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                continue

            target_nodes = solution[:, target_step, 0]
            target_flags = solution[:, target_step, 1]

            current_node = solution[:, target_step - 1, 0]

            if torch.any(target_nodes >= problem_size) or torch.any(target_nodes < 0):
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                continue

            # Calculate remaining capacity for this MTP depth by simulating intermediate steps
            depth_remaining_capacity = self._calculate_remaining_capacity_for_depth(
                remaining_capacity, solution, current_step, depth, problems
            )

            # Normalize remaining capacity
            remaining_capacity_norm = depth_remaining_capacity / self.capacity  # [B]

            if prev_repr is None:
                # First depth: use base representation
                prev_repr = base_node_repr

            # current_embedding = self._get_node_embeddings(encoded_nodes, current_node)
            current_embedding = self._get_node_embeddings(encoded_nodes, current_node)

            # Pass all required inputs to MTP module including capacity information
            mtp_repr = self.mtp_modules[depth - 1](
                prev_repr,
                current_embedding,
                remaining_capacity_norm,
                selected_node_list,
                depth
            )


            mtp_logits = self._generate_mtp_probabilities(
                mtp_repr, encoded_nodes, selected_node_list, self.mtp_modules[depth - 1]
            )

            if torch.all(torch.isinf(mtp_logits)):
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                continue

            is_via_depot = target_flags == 1
            target_indices = target_nodes.clone() - 1
            target_indices[is_via_depot] += (problem_size - 1)

            try:
                if torch.any(target_indices >= mtp_logits.shape[1]) or torch.any(target_indices < 0):
                    mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                    continue

                if torch.all(torch.isinf(mtp_logits)) or torch.all(torch.isnan(mtp_logits)):
                    mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)
                    continue

                mtp_loss = F.cross_entropy(
                    mtp_logits, target_indices.long(), reduction='mean'
                )

                if torch.isnan(mtp_loss) or torch.isinf(mtp_loss) or mtp_loss <= 0:
                    mtp_loss = torch.tensor(0.0, device=encoded_nodes.device)

                mtp_losses[f'mtp_depth_{depth}'] = mtp_loss
            except Exception as e:
                print(f"MTP loss computation failed for depth {depth}: {e}")
                mtp_losses[f'mtp_depth_{depth}'] = torch.tensor(0.0, device=encoded_nodes.device)

            # Update prev_repr for next depth
            prev_repr = mtp_repr

        return mtp_losses

    def _get_embedding_from_decoder_out(self, decoder_out, nodes_to_get, selected_node_list, problem_size):
        """Get embeddings for specific nodes from decoder_out"""
        batch_size = decoder_out.shape[0]
        embedding_dim = decoder_out.shape[2]
        num_customers = problem_size - 1

        # Determine unselected nodes at the time of decoder_out creation
        selected_node_list_0based = selected_node_list.clone().detach() - 1
        all_customers = torch.arange(num_customers, device=decoder_out.device)[None, :].expand(batch_size, -1)

        # Create a mask for selected nodes
        selection_mask = torch.zeros_like(all_customers, dtype=torch.bool)
        if selected_node_list_0based.shape[1] > 0:
            batch_indices = torch.arange(batch_size, device=decoder_out.device)[:, None]
            selection_mask.scatter_(1, selected_node_list_0based, True)

        # `nodes_to_get` are 1-based, convert to 0-based customer indices
        nodes_to_get_0based = nodes_to_get - 1

        # Vectorized rank calculation
        # 1. Get a mask of unselected nodes
        unselected_mask = ~selection_mask
        # 2. Calculate the cumulative sum of unselected nodes. This gives the count of available nodes up to each index.
        cumulative_unselected = torch.cumsum(unselected_mask.int(), dim=1)
        # 3. Gather the cumulative count for the specific nodes we want to get.
        target_cumulative_counts = cumulative_unselected.gather(dim=1, index=nodes_to_get_0based[:, None])
        # 4. The rank is the cumulative count minus 1.
        rank = (target_cumulative_counts - 1).squeeze(1)

        # `decoder_out` has [depot, available_nodes, last_node], so add 1 for the depot
        gathering_index = (1 + rank)[:, None, None].expand(-1, 1, embedding_dim)

        # Gather embeddings from decoder_out
        node_embeddings = decoder_out.gather(dim=1, index=gathering_index.long()).squeeze(1)

        return node_embeddings

    def _calculate_remaining_capacity_for_depth(self, current_remaining_capacity, solution, current_step, depth,
                                                problems):
        """Calculate remaining capacity after visiting intermediate nodes up to the target depth"""
        if problems is None:
            # Fallback: return current remaining capacity if problems data not available
            return current_remaining_capacity.clone()

        batch_size = current_remaining_capacity.shape[0]
        depth_remaining_capacity = current_remaining_capacity.clone()

        # Subtract demands from intermediate steps
        for step_offset in range(1, depth):
            intermediate_step = current_step + step_offset
            if intermediate_step < solution.shape[1]:
                intermediate_nodes = solution[:, intermediate_step, 0]  # [B]
                intermediate_flags = solution[:, intermediate_step, 1]  # [B]

                # Only subtract demand if not returning to depot (flag == 0)
                # When flag == 1, vehicle returns to depot and capacity is reset
                non_depot_mask = intermediate_flags == 0

                if torch.any(non_depot_mask):
                    # Get demands for intermediate nodes from problems[:, :, 2]
                    # Convert 1-based node indices to 0-based for indexing
                    node_indices = intermediate_nodes - 1

                    # Handle valid node indices only
                    valid_indices = (node_indices >= 0) & (node_indices < problems.shape[1]) & non_depot_mask

                    if torch.any(valid_indices):
                        batch_indices = torch.arange(batch_size, device=problems.device)[valid_indices]
                        valid_node_indices = node_indices[valid_indices]

                        # Get demands for valid nodes
                        demands = problems[batch_indices, valid_node_indices, 2]

                        # Subtract demands from remaining capacity
                        depth_remaining_capacity[valid_indices] -= demands

                        # Ensure non-negative capacity
                        depth_remaining_capacity = torch.clamp(depth_remaining_capacity, min=0.0)

                # Reset capacity to full when returning to depot
                depot_return_mask = intermediate_flags == 1
                if torch.any(depot_return_mask):
                    depth_remaining_capacity[depot_return_mask] = self.capacity

        return depth_remaining_capacity

    def _get_main_model_representation(self, encoded_nodes, selected_node_list):
        """Get node representation from main model for MTP input (without capacity transformation)"""
        batch_size = encoded_nodes.shape[0]
        embedding_dim = encoded_nodes.shape[2]

        if selected_node_list.shape[1] == 0:
            embedded_node = encoded_nodes[:, 0, :]
        else:
            last_selected = selected_node_list[:, -1]
            gathering_index = last_selected[:, None, None].expand(batch_size, 1, embedding_dim)
            embedded_node = encoded_nodes.gather(dim=1, index=gathering_index).squeeze(1)

        return embedded_node

    def _get_node_embeddings(self, encoded_nodes, node_indices):
        """Get embeddings for specific nodes from encoded_nodes"""
        batch_size = encoded_nodes.shape[0]
        embedding_dim = encoded_nodes.shape[2]

        # Convert 1-based node indices to 0-based for gathering
        node_indices_0based = node_indices - 1
        
        # Create gathering indices
        gathering_index = node_indices_0based[:, None, None].expand(batch_size, 1, embedding_dim)
        
        # Get embeddings for the specified nodes
        node_embeddings = encoded_nodes.gather(dim=1, index=gathering_index).squeeze(1)
        
        return node_embeddings

    def _generate_mtp_probabilities(self, mtp_repr, decoder_out, selected_node_list, mtp_module):
        """Generate probability distribution using decoder-based approach with MTP representation"""
        batch_size = mtp_repr.shape[0]
        seq_length = decoder_out.shape[1]
        embedding_dim = decoder_out.shape[2]
        
        # Clone decoder_out to avoid modifying the original
        modified_decoder_out = decoder_out.clone()

        # modified_decoder_out[:, -1, :] = mtp_repr
        modified_decoder_out = torch.cat([modified_decoder_out, mtp_repr.unsqueeze(1)], dim=1)  # [B, seq_length + 1, embedding_dim]
        
        # Apply decoder layer for self-attention
        decoder_layer_out = mtp_module.mtp_decoder_layer(modified_decoder_out)
        
        # Apply MTP-specific Linear_final to get logits
        position_logits = mtp_module.mtp_Linear_final(decoder_layer_out)  # [B, seq_length, 2]
        
        # Apply masking to first and last nodes (same as main decoder)
        position_logits[:, [0, -1, -2], :] = position_logits[:, [0, -1, -2], :] + float('-inf')
        
        # Flatten the logits following the main decoder's approach
        props = torch.cat((position_logits[:, :, 0], position_logits[:, :, 1]), dim=1)  # [B, 2*seq_length]
        
        # Apply softmax to convert to probabilities
        props = F.softmax(props, dim=-1)
        
        # Extract customer probabilities (excluding first and last nodes)
        customer_num = seq_length - 2  # Available nodes = total - first - last
        customer_props = torch.cat((props[:, 1:customer_num + 1], props[:, seq_length + 1:-1]), dim=1)
        
        # Determine problem size
        problem_size = customer_num + 1  # Add depot
        
        # Create full probability tensor with proper masking
        new_props = torch.zeros(batch_size, 2 * customer_num, device=decoder_out.device)
        
        # Apply masking for already selected nodes (similar to main decoder)
        if selected_node_list.shape[1] > 0:
            selected_node_list_ = selected_node_list.clone().detach() - 1  # Convert to 0-based
            
            # Start with customer_props as base
            new_props = customer_props.clone()
            
            # Create batch indices for masking
            batch_indices = torch.arange(batch_size, dtype=torch.long, device=decoder_out.device)[:, None]
            
            # Mask direct visits and via-depot visits
            direct_indices = selected_node_list_
            depot_indices = customer_num + selected_node_list_
            
            # Apply masks for valid indices
            valid_indices = (direct_indices >= 0) & (direct_indices < customer_num)
            if torch.any(valid_indices):
                batch_idx = batch_indices.expand_as(selected_node_list_)[valid_indices]
                node_idx = direct_indices[valid_indices]
                new_props[batch_idx, node_idx] = 1e-7
                
                # Make sure depot indices are within bounds
                depot_idx = depot_indices[valid_indices]
                valid_depot = depot_idx < new_props.shape[1]
                if torch.any(valid_depot):
                    new_props[batch_idx[valid_depot], depot_idx[valid_depot]] = 1e-7
        else:
            new_props = customer_props
        
        # Ensure numerical stability
        new_props = torch.clamp(new_props, min=1e-7)
        
        # Normalize to ensure valid probabilities
        new_props = new_props / new_props.sum(dim=1, keepdim=True)

        return new_props


class MTP_Module(nn.Module):
    """Multi-Token Prediction Module for predicting future nodes in CVRP using decoder output"""

    def __init__(self, depth, **model_params):
        super().__init__()
        self.depth = depth
        embedding_dim = model_params['embedding_dim']

        # Capacity embedding dimension
        self.capacity_emb_dim = embedding_dim // 4

        # Input projection for combined representation (prev_repr + target_embedding + first_node + capacity)
        input_dim = embedding_dim * 2 + self.capacity_emb_dim
        self.input_projection = nn.Linear(input_dim, embedding_dim, bias=True)
        # self.input_norm = nn.LayerNorm(embedding_dim)
        self.input_norm = nn.RMSNorm(embedding_dim)

        # Capacity processing layer
        self.capacity_embedding = nn.Linear(1, self.capacity_emb_dim, bias=True)

        # Enhanced transformer block for capacity-aware processing
        self.transformer_block = MTP_TransformerBlock(**model_params)
        
        # Additional layer for processing decoder output
        self.decoder_projection = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        
        # Add decoder layer for self-attention based probability generation
        self.mtp_decoder_layer = DecoderLayer(**model_params)
        
        # MTP-specific Linear_final layer for logit generation
        self.mtp_Linear_final = nn.Linear(embedding_dim, 2, bias=True)


    def forward(self, prev_repr, target_embedding, remaining_capacity_norm, selected_node_list,
                depth):
        """
        Forward pass with capacity information

        Args:
            prev_repr: Previous representation from MTP or base model
            target_embedding: Target node embedding
            embedded_first_node: Depot/first node embedding
            remaining_capacity_norm: Normalized remaining capacity
            selected_node_list: List of selected nodes
            depth: Current prediction depth
        """
        batch_size = prev_repr.shape[0]

        # Process capacity information
        capacity_emb = self.capacity_embedding(remaining_capacity_norm.unsqueeze(-1))  # [B, capacity_emb_dim]



        # Combine all components
        combined_input = torch.cat([
            self.input_norm(prev_repr),
            self.input_norm(target_embedding),
            capacity_emb
        ], dim=-1)  # [B, embedding_dim * 3 + capacity_emb_dim]

        # Project to embedding dimension
        projected_input = self.input_projection(combined_input)

        # Apply transformer block
        mtp_repr = self.transformer_block(projected_input.unsqueeze(1)).squeeze(1)

        return mtp_repr


class MTP_TransformerBlock(nn.Module):
    """Transformer block used in MTP modules"""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=head_num,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = 1
        self.embedding = nn.Linear(3, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data_, capacity):
        data = data_.clone().detach()
        data = data[:, :, :3]
        data[:, :, 2] = data[:, :, 2] / capacity
        embedded_input = self.embedding(data)
        out = embedded_input

        layer_count = 0
        for layer in self.layers:
            out = layer(out)
            layer_count += 1
        return out


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


class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.Linear_final = nn.Linear(embedding_dim, 2, bias=True)

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

    def forward(self, data, selected_node_list, capacity, remaining_capacity, return_decoder_out=False):
        data_ = data[:, 1:, :].clone().detach()
        selected_node_list_ = selected_node_list.clone().detach() - 1

        batch_size_V = data_.shape[0]
        problem_size = data_.shape[1]
        new_data = data_.clone().detach()

        left_encoded_node = self._get_new_data(new_data, selected_node_list_, problem_size, batch_size_V)

        embedded_first_node = data[:, [0], :]

        if selected_node_list_.shape[1] == 0:
            embedded_last_node = data[:, [0], :]
        else:
            embedded_last_node = self._get_encoding(new_data, selected_node_list_[:, [-1]])

        remaining_capacity = remaining_capacity.reshape(batch_size_V, 1, 1) / capacity
        first_node_cat = torch.cat((embedded_first_node, remaining_capacity), dim=2)
        last_node_cat = torch.cat((embedded_last_node, remaining_capacity), dim=2)

        embedded_first_node_ = self.embedding_first_node(first_node_cat)
        embedded_last_node_ = self.embedding_last_node(last_node_cat)

        embeded_all = torch.cat((embedded_first_node_, left_encoded_node, embedded_last_node_), dim=1)
        out = embeded_all

        layer_count = 0
        for layer in self.layers:
            out = layer(out)
            layer_count += 1

        # Store decoder output before Linear_final for MTP
        decoder_out_before_final = out.clone()

        out = self.Linear_final(out)
        out[:, [0, -1], :] = out[:, [0, -1], :] + float('-inf')
        out = torch.cat((out[:, :, 0], out[:, :, 1]), dim=1)

        props = F.softmax(out, dim=-1)
        customer_num = left_encoded_node.shape[1]

        props = torch.cat((props[:, 1:customer_num + 1], props[:, customer_num + 1 + 1 + 1:-1]),
                          dim=1)

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)
        props = props_clone

        new_props = torch.zeros(batch_size_V, 2 * (problem_size))

        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].repeat(1, selected_node_list_.shape[1] * 2)
        index_2_ = torch.cat(
            ((selected_node_list_).type(torch.long), (problem_size) + (selected_node_list_).type(torch.long)), dim=-1)
        new_props[index_1_, index_2_,] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        new_props[index] = props.ravel()

        if return_decoder_out:
            return new_props, decoder_out_before_final
        else:
            return new_props


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