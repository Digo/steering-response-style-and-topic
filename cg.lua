require 'nn'
local CG = torch.class('CountingGrid')

function CG:__init(E)
	self.E = E
	self.L = torch.prod(E)
    self.minp = 1/(10000000*self.L)
end

function CG:map2grid(counts)
	local Z = self.logh:size(1)
	local lql = torch.mm(self.logh:view(Z,self.L):t(), counts)  -- L,Z Z,T
    local Q = torch.zeros(self.L, counts:size(2)):typeAs(self.logh)
	CG.calculate_Q(Q, lql, self.minp)
	return Q
end

function CG:fit(counts, W, options)
	torch.setdefaulttensortype(options.dtype)
  	
	local E = self.E
	local L = self.L
	local Z, T, pseudocounts

	local sparse_counts, sparse_counts_t
	if options.sparse == 1 then
		sparse_counts = counts.sparse_counts:type(options.dtype)
		sparse_counts_t = counts.sparse_counts_t:type(options.dtype)
		Z = sparse_counts[-1][1]
		T = sparse_counts_t[-1][1]
		pseudocounts = sparse_counts[{{},3}]:sum() / (2.5 * L * T)
	else
		Z = counts:size(1)
		T = counts:size(2)
		if options.normalize_data == 1 then --TODO create sparse version as well
			counts:cdiv(torch.sum(counts,1):expand(Z,T)):mul(100*torch.prod(W))
		end
    	pseudocounts = counts:sum() / (2.5 * L * T) 
		counts = counts:type(options.dtype)
	end

	local ww = W:prod()
    
	local pl = options.pl or torch.ones(E[1],E[2]):div(L)
    
    local pi = options.pi
    if not pi then
		pi = 1+1*torch.rand(Z,E[1],E[2])
        pi = pi:cdiv(pi:sum(1):expand(Z,E[1],E[2]))
    end
   
    local pi_buf = torch.Tensor(Z, E[1]+W[1]-1, E[2]+W[2]-1)
    pi = pi_buf[{{},{1,E[1]},{1,E[2]}}]:copy(pi)
    CG.circular_pad(pi_buf,E,0,0,W[1]-1,W[2]-1)

    local avg_pooling = nn.SpatialAveragePooling(W[1], W[2], 1, 1)
    local h = avg_pooling:forward(pi_buf)
    local logh = torch.log(h)

    local alpha = 1e-10
	local ww_alpha = ww*alpha
    local start_iterating_m = 1  --Start M-step iterations from
    local m_step_iter = 1 -- M-step iterations: fasten convergence
    
    
    local loglikelihood = torch.zeros(options.max_iter)
   
    local Lq
    if options.monly == 1 then
     Lq = options.Lq:view(L,T)
    end
    local Q = torch.zeros(L,T)
	
	misc.print_memory_usage(true, 0)
    
    local QH_buf = torch.Tensor(Z, E[1]+W[1]-1, E[2]+W[2]-1)
    local QH = QH_buf[{{},{W[1],E[1]+W[1]-1},{W[2],E[2]+W[2]-1}}]
			
	local nrm = torch.Tensor(Z, L):cuda() -- Z,W,W
	local zero_bias = torch.zeros(L):cuda()
		
	local lql 
	if options.sparse == 1 then
		local logh_t = logh:view(Z,L):t():contiguous()
		lql = torch.Tensor(T,L):cuda()
		sparse_counts.THNN.SparseLinear_updateOutput(
			sparse_counts_t:cdata(),
			lql:cdata(),
			logh_t:cdata(),
			zero_bias:cdata()
		)
		lql = lql:t()
	else
		lql = torch.mm(logh:view(Z,L):t(), counts)  -- L,Z Z,T
	end
	misc.print_memory_usage(true, 0)
    
    for iter=1,options.max_iter do
		local timer = torch.Timer()
		if options.monly ~= 1 then
			if options.learn_pl == 1 then
				print('TODO: not implemented.')
			end
			Lq = CG.calculate_Q(Q, lql, self.minp)
		end
		local qLq = Lq:cmul(Q):sum() -- Lq contaminated

		if options.sparse == 1 then
			nrm = nrm:view(Z,L)
			sparse_counts.THNN.SparseLinear_updateOutput(
				sparse_counts:cdata(),
				nrm:cdata(),
				Q:cdata(),
				zero_bias:cdata()
			)
			nrm = nrm:view(Z,E[1],E[2])
		else
			nrm = nrm:view(Z,L):mm(counts, Q:t()):view(Z, E[1],E[2])
		end
        
		local miter = (iter > start_iterating_m) and m_step_iter or 1
        for _ = 1,miter do
            if options.learn_pi then
				QH:cdiv(nrm, torch.add(h, ww_alpha))
                CG.circular_pad(QH_buf,E,W[1]-1,W[2]-1,0,0)
				local QH_sum = avg_pooling:forward(QH_buf):mul(ww) -- QH_sum's storage will be contaminated by h
                QH_sum[QH_sum:lt(0)] = 0  -- Z,W,W
                
                local un_pi = QH_sum:cmul(pi:add(alpha)):add(pseudocounts)
                pi:cdiv(un_pi, un_pi:sum(1):expandAs(un_pi))
                
                CG.circular_pad(pi_buf,E,0,0,W[1]-1,W[2]-1)
                h = avg_pooling:forward(pi_buf)
            end
        end
		logh:log(h)

		if options.sparse == 1 then
			local logh_t = logh:view(Z,L):t():contiguous()
            lql = lql:resize(T,L)
			sparse_counts.THNN.SparseLinear_updateOutput(
				sparse_counts_t:cdata(),
				lql:cdata(),
				logh_t:cdata(),
				zero_bias:cdata()
			)
			lql = lql:t()
		else
            torch.mm(lql, logh:view(Z,L):t(), counts)  -- L,Z Z,T
		end

		local loglikelihood_samples = Q:cmul(lql):sum() - qLq
		loglikelihood[iter] = loglikelihood_samples
        print(string.format("%3d iteration:  loglikelihood= %6.0f   time/batch= %.3fs ", iter, loglikelihood[iter], timer:time().real))
     
        if iter > 30 then
            local F1 = loglikelihood[iter]    --/total
            local F2 = loglikelihood[iter-1]  --/total
            local rel_ch = 2 * (F1-F2) / (math.abs(F1) + math.abs(F2))
            if rel_ch < options.min_change then break end
        end
		if iter % 5 == 0 then collectgarbage() end
    end
	
	--update 
	self.pi = pi
	self.logh = logh
   
	return pi,pl,Lq
end

-- inplace update to Q
function CG.calculate_Q(Q, lql, minp)
	local lql_sub_max = lql:csub(lql:max(1):expandAs(lql))
	local lql_sub_max_sum = torch.exp(lql_sub_max):sum(1):log():expandAs(lql_sub_max)
	local Lq = lql_sub_max:csub(lql_sub_max_sum)
	Q:exp(Lq)
	Q[Q:lt(minp)] = minp
	Q:cdiv(Q:sum(1):expandAs(Q)) 
	Lq:log(Q)
	return Lq
end

-- pad_x must >= 0, only tl and bt 
function CG.circular_pad(buf, E, pad_t, pad_l, pad_b, pad_r)
    local h = E[1] + pad_t + pad_b
    local w = E[2] + pad_l + pad_r
    if pad_t > 0 then buf[{{},{1,pad_t},{1,w}}]:copy(buf[{{},{h-pad_t+1,h},{1,w}}]) end
    if pad_l > 0 then buf[{{},{1,h},{1,pad_l}}]:copy(buf[{{},{1,h},{w-pad_l+1,w}}]) end
    if pad_b > 0 then buf[{{},{h-pad_b+1,h},{1,w}}]:copy(buf[{{},{1,pad_b},{1,w}}]) end
    if pad_r > 0 then buf[{{},{1,h},{w-pad_r+1,w}}]:copy(buf[{{},{1,h},{1,pad_r}}]) end 
end

function CG.sparse_2ways(counts)
	local data = {}
	data.sparse_counts = CG.sparse(counts)
	data.sparse_counts_t = CG.sparse(counts:t())
	return data
end

-- still slow, not optimized 
function CG.sparse(counts)
	local non_zero_idx = counts:nonzero()
	local sparse_counts = torch.LongTensor():resize(non_zero_idx:size(1),3)
	sparse_counts[{{},{1,2}}] = non_zero_idx
	sparse_counts[{{},{3}}] =  counts[counts:ne(0)]
	return sparse_counts
end

function CG.sparse_transpose(wd)
	local _, s_index = wd[{{},2}]:sort()	
	local s_val = wd:index(1,s_index) --alloc new storage
	local cur = 0
	local start = 1
	local y_values = s_val[{{},2}]
	local val = y_values[1]
	y_values:apply(function(x)  
		cur = cur + 1
		if x ~= val then
			local same_y = s_val[{{start,cur-1}}]
			local _,sortx_i = same_y[{{},1}]:sort(1)
			same_y:copy(same_y:index(1,sortx_i))
			start = cur
			val = x
		end
	end)
	local same_y = s_val[{{start,cur}}]
	local _,sortx_i = same_y[{{},1}]:sort()
	same_y:copy(same_y:index(1,sortx_i))

	local s_val_t = s_val:clone()
	s_val_t[{{},1}] = s_val[{{},2}]
	s_val_t[{{},2}] = s_val[{{},1}]
	return s_val_t
end

return CG
