from math import sqrt
import numpy as np
import scipy.sparse as sp
import time
from gurobipy import *

from utils import Vts_to_Pts_sparse

# Prune active set 
def prune(Pts, Vts, psi_q):
    o = Model()
    lamb = o.addMVar(Pts.shape[1], lb = 0, ub = 1, vtype = GRB.CONTINUOUS)
    o.update()
    o.Params.OutputFlag = 0
    o.addConstr(lamb.sum() == 1)
    simcon = o.addConstr(psi_q == Pts @ lamb)
    o.optimize()
        
    if o.status == GRB.INFEASIBLE:
        simcon = simcon.tolist()
        o.feasRelax(0, False, None, None, None, simcon, np.ones(len(simcon)))
        o.optimize()
    
    return(Pts[:,lamb.X > 0], Vts[:,lamb.X > 0], lamb.X[lamb.X > 0][:,None])

# Golden section search
def golden(X, Y, psi_q, psi_n, lamb_q, lamb_n, lpar):
    n = X.shape[0]
    dpsi = psi_q[X] - psi_n[X]
    gam = np.dot(Y/lpar-psi_n[X],dpsi)/np.dot(dpsi,dpsi)
    
    if (gam < 0):
        gam = 0
    elif (gam > 1):
        gam = 1
        
    return(gam*psi_q + (1-gam)*psi_n, gam*lamb_q + (1-gam)*lamb_n, gam)

# Callback function to make Gurobi "lazy"
def callback(model, where):
    if where == GRB.Callback.MIP:
        model._gap = np.minimum(model._gap, (model._cmin - model.cbGet(GRB.Callback.MIP_OBJBND))/2)
        if (model._cmin - model.cbGet(GRB.Callback.MIP_OBJBST) > model._gap):
            model._oracle = "LazyIP"
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        model._gap = np.minimum(model._gap, (model._cmin - model.cbGet(GRB.Callback.MIPSOL_OBJBND))/2)
        if (model._cmin - model.cbGet(GRB.Callback.MIPSOL_OBJ) > model._gap):
            model._oracle = "LazyIP"
            model.terminate()
    elif where == GRB.Callback.MIPNODE:
        model._gap = np.minimum(model._gap, (model._cmin - model.cbGet(GRB.Callback.MIPNODE_OBJBND))/2)
          
# Alternating minimization oracle
# @profile 
def altmin(r, lpar, p, tol, cmin, gap, c, the_q, Un, indices=False, pattern=False, altmin_pattern=None):
    """
    indices (bool): implement the fpro computation speed up if True
    pattern (np.array): implement the pattern speed up based on the fpro computation speed up if not None
    """
    if altmin_pattern is not None:
        assert indices, 'the pattern speed up must be used with the fpro computation speed up'
    the = the_q.copy()
    cum_r = np.insert(np.cumsum(r), 0, 0)
    un = Un.shape[0]

    last_cmin = cmin + 1e6
    cnt = 0
    lpar_c = lpar*c
    while (True):
        cnt += 1
        for ind in range(p):
            # fpro is the c_k when converting <theta_1 x theta_2 x ... x theta_p, grad(f(psi_q))>
            # to <theta_ind, c_k>. Conversion is done between lines 69 to 75
            # fpro is the pro transformed to have the same shape as theta_ind
            # pro has the same shape as grad(f(psi_q)). 
            # pro has the same information as fpro
            pro = lpar_c.copy()
            for k in range(p):
                if (ind != k):
                    pro = np.multiply(pro, the[cum_r[k] + Un[:,k]]) 

            fpro = np.zeros(r[ind])

            if indices:
                for i in range(r[ind]):
                    if pattern:
                        fpro[i] += pro[altmin_pattern[ind, i]].sum()
                    else:
                        fpro[i] += pro[np.argwhere(Un[:, ind] == i)].sum()
            else:
                for k in range(un):
                    fpro[Un[k,ind]] += pro[k]

            the[cum_r[ind]:cum_r[ind+1]] = (fpro < 0).astype(int)
            curr_cmin = np.sum(fpro[fpro < 0])

        if (curr_cmin > last_cmin - tol):
            break
        else:
            last_cmin = curr_cmin

    psi = np.ones(un)
    for k in range(p):
        psi = np.multiply(psi, the[cum_r[k] + Un[:,k]])

    return(psi, the, curr_cmin)

# (Khatri-Rao) Alternating minimization completion
def krcomp(X, Y, r, rank, lpar = 1, tol = 1e-6, verbose = True):

    # Compute derived parameters
    n = X.shape[0]
    p = len(r)
    cum_r = np.insert(np.cumsum(r), 0, 0)

    if (len(X.shape) == 1):
        # Convext X into wide indices
        wideX = False
        Xo = X.copy()
        X = np.array(np.unravel_index(Xo, r)).T+1
    else:
        # X already has wide indices
        wideX = True

    # Initialize bookkeeping variables
    prnt_count = 0
    last_objVal = 1e6
    cnt = 0
    
    # Best point to date
    the_q = np.random.uniform(0,np.max(Y)**(1/p),((np.sum(r), rank)))

    # Setup timer
    elapsed_time = 0
    last_time = time.process_time()
    last_time = time.time()
    prnt_time = last_time - 10
    
    # Run alternating minimization
    the = the_q.copy()
    is_true = True
    while is_true:
        cnt += 1
        for ind in range(p):
            A = np.zeros((n+rank*r[ind], rank*r[ind]))
            A[n:(n+rank*r[ind]),:] = np.sqrt(lpar)*np.eye(rank*r[ind])
            b = np.hstack((Y, np.zeros((rank*r[ind],))))

            fflag = True
            for ink in range(p):
                if (ink == ind):
                    continue
                
                if (fflag):
                    for inr in range(rank):
                        A[np.arange(n), inr*r[ind]+X[:,ind]-1] = the[cum_r[ink]+X[:,ink]-1,inr]
                    fflag = False
                else:
                    for inr in range(rank):
                        A[np.arange(n), inr*r[ind]+X[:,ind]-1] *= the[cum_r[ink]+X[:,ink]-1,inr]
                
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            for inr in range(rank):
                the[cum_r[ind]:cum_r[ind+1],inr] = x[inr*r[ind]:(inr+1)*r[ind]]
            #res = A[0:n,:] @ x - b[0:n]
            res = A @ x - b
            objVal = np.dot(res,res)/n
                
        if (objVal > last_objVal - tol):
            is_true = False
            
        if (verbose & (prnt_count % 20 == 0)):
            print("")
            print("   Objective   |   Iters   Time")
            print("")

        curr_time = time.process_time()
        curr_time = time.time()
        elapsed_time = curr_time - last_time
        if (verbose & (prnt_count % 20 == 0 or not is_true or prnt_time <= curr_time - 5)):
            prnt_time = curr_time
            prnt_count += 1
            print("%12.3e     %6i %6is" % 
                (objVal, cnt, elapsed_time))
        
        last_objVal = objVal
    
    if verbose:
        print("\n")
        print("Solution found (tolerance %7.2e)" % (tol))
        print("Best objective %10.6e" % (objVal))

    psi_q = np.zeros(np.prod(r))
    for ind in range(rank):
        the_n = the[cum_r[0]:cum_r[1],ind]
        for k in range(1,p):
            the_n = np.tensordot(the_n, the[cum_r[k]:cum_r[k+1],ind], axes = 0)

        psi_q += the_n.flatten()
        
    if not wideX:
        X = Xo
        
    return(psi_q)

# @profile
def nonten(X, Y, r, lpar = 1, tol = 1e-6, verbose = True, indices=False, pattern=False, sparse=False, nag=False):
    """
    X: (n, ) the indices of known entries in the flatten version of the true tensor
    Y: (n, ) values of known entries corresponding to the indices in X
    r: (r_1, ..., r_p) dimension of the truth tensor
    lpar: lambda parameter in eqn (8)
    indices (bool): implement the fpro computation speed up in altmin if True
    pattern (bool): implement the pattern speed up based on the fpro computation speed up in altmin if True
    sparse (bool): implement the scipy sparse matrix speed up if True
    nag (bool): implement the Nesterov's accelerated gradient descent
    """
    if pattern:
        assert indices, 'the pattern speed up must be used with the fpro computation speed up'

    # Setup timer
    elapsed_time = 0
    last_time = time.process_time()
    last_time = time.time()
    prnt_time = last_time - 10

    # Compute derived parameters
    n = X.shape[0]
    p = len(r)
    cum_r = np.insert(np.cumsum(r), 0, 0)
    
    if (len(X.shape) == 1):
        # X already has flat indices
        wideX = False
    else:
        # Convert X into flat indices
        wideX = True
        Xo = X.copy()
        X = np.ravel_multi_index((Xo.T-1).tolist(), r)

    # Variables for projected polytope 
    # see last two sentences in the secend paragraph of the section 4.3
    [uinds, ucnt] = np.unique(X, return_counts=True) # uinds: (sorted) unique indices of known entries
    un = len(uinds) # unique number of known entries
    Un = np.zeros((un, p), dtype=int) # unique coordinate indices of known entries
    Xn = np.zeros(n, dtype=int) # i-th element is the index of the i-th sample
    X_uni = np.unique(X)
    imup = un/(2*lpar*np.max(ucnt)/n)
    
    # Variables for the linearized optimization problem
    m = Model()
    #E3m.Params.OutputFlag = 0
    var = m.addMVar(int(un + np.sum(r)), vtype = GRB.BINARY)
    psi = var[0:un] # variable in FW, the direction to move along with
    psi.setAttr('VType', 'c')
    psi.setAttr('UB', 1)
    psi.setAttr('LB', 0)
    the = var[un:un+np.sum(r)]
    m.update()
    m.Params.Method = 2

    # Constraints are:
    #     un elements each with (phi_x)
    #     p upper bound constraints (RHS of the 2nd row in the eqn (13))
    #     1 lower bound constraint (LHS of the 2nd and the 1st row in the eqn (13))
    #
    # Upper bound constraint has:
    #     2 variables each (one phi_x and one theta_x_k)
    #
    # Lower bound constraint has:
    #     (p+1) variables each (one phi_x and p theta_x_k)
    #
    # Data matrix for constructing sparse matrix has:
    #     (2*p+(p+1))*un rows
    #
    # Constraint matrix has:
    #     (p+1)*un rows
    #     un + np.sum(r) cols

    # Constraints for linearized optimization problem 
    # Here we project C_1 onto U at the same time
    data = np.zeros((2*p+(p+1))*un)
    row_ind = np.zeros((3*p+1)*un)  # row indices of non-zero entries
    col_ind = np.zeros((3*p+1)*un)  # col indices of non-zero entries

    ind_vec = np.zeros(p, dtype=int)
    for cnt in range(un):
        Xn[X == uinds[cnt]] = cnt 
        ind_vec = np.unravel_index(uinds[cnt], r)
        Un[cnt,:] = ind_vec
        
        # build the "p upper bound constraints"
        for k in range(p):
            data[2*p*cnt+2*k] = 1
            row_ind[2*p*cnt+2*k] = p*cnt + k
            col_ind[2*p*cnt+2*k] = cnt

            data[2*p*cnt+2*k+1] = -1
            row_ind[2*p*cnt+2*k+1] = p*cnt + k
            col_ind[2*p*cnt+2*k+1] = un + cum_r[k] + ind_vec[k]

        # build the "1 lower bound constraint" - the first row in eqn 13
        data[2*p*un+(p+1)*cnt] = -1
        row_ind[2*p*un+(p+1)*cnt] = p*un + cnt
        col_ind[2*p*un+(p+1)*cnt] = cnt

        # build the "1 lower bound constraint" - the first row in eqn 13
        data[2*p*un+(p+1)*cnt+1:2*p*un+(p+1)*cnt+p+1] = 1
        row_ind[2*p*un+(p+1)*cnt+1:2*p*un+(p+1)*cnt+p+1] = p*un + cnt
        col_ind[2*p*un+(p+1)*cnt+1:2*p*un+(p+1)*cnt+p+1] = un + cum_r[0:p] + ind_vec[0:p] 

    A = sp.csr_matrix((data, (row_ind, col_ind)), shape=((p+1)*un, un+np.sum(r)))
    b = np.zeros((p+1)*un)
    b[p*un+0:p*un+un] = p-1

    m.addConstr(A @ var <= b) # the matrix form of the constraints in eqn (13)

    # m.write('nonten.lp')

    # Initialize bookkeeping variables
    iter_count = 0
    prnt_count = 0 # print count
    sigd_count = 0 # simplex gradient descent (in BCG paper)
    orcl_count = 0 # oracle count
    ip_count = 0 # integer program count 
    bestbd = 0 # best lower bound for obj func in eqn 8
    as_drops = 0 # active set drop (in BCG paper)
    m._gap = float('inf') # Phi_0 in Line 1, but we haven't really initialized the gap estimate until the Integer LP at the first iteration
    last_gap = float('inf')

    # Best point to date
    # Initialization
    Vts = np.ones((np.sum(r),1)) # (non-projected) active vertex set (S_t in Line 2). Elements (theta which can be used to recover psi) are the vertices of C_1
    if sparse:
        Pts = Vts_to_Pts_sparse(Vts, X_uni, r, cum_r) # (projected) active vertex set (Proj_U(S_t) in Line 2). Elements (psi) are the vertices of Proj_U(C_1)
    else:
        Pts = np.ones((un,1))
    psi_q = np.ones(un) # the current iterate x_t (in Proj_U(C_1)) which is a cvx combination of elements in Pts using lamb as the coefficients
    the_q = np.ones(np.sum(r))
    lamb = np.array([[1]]) # convex comb coefficients to get the current iterate

    # Initialization for Nesterov’s Accelerated Gradient Descent (NAG) (https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/)
    if nag:
        psi_last = psi_q # y_1 in NAG
        lam_0 = 0
        lam_s = (1 + sqrt(1 + 4 * (lam_0 ** 2))) / 2
        lam_s_plus_1 = (1 + sqrt(1 + 4 * (lam_s ** 2))) / 2
        gam_s = (1 - lam_s) / lam_s_plus_1

    # pattern for altmin
    altmin_pattern = None
    if pattern:
        altmin_pattern = np.zeros((p, max(r)), dtype=np.object)
        for ind in range(p):
            for i in range(r[ind]):
                altmin_pattern[ind, i] = np.argwhere(Un[:, ind] == i)

    ### BCG ###
    is_true = True
    while is_true:
        iter_count += 1
        # calculate linearized cost
        c = np.zeros(un) # partial derivatives of the obj function w.r.t. each known entry. 
        for ind in range(n):
            c[Xn[ind]] += -2/n*(Y[ind] - lpar*psi_q[Xn[ind]]) # assign the derivative w.r.t. the known entry with the i-th flatten index of the tensor to the i-th element
        lpar_c = lpar*c

        if sparse:
            pro = Pts.T.dot(lpar_c) # grad(f(x_t))v
        else:
            pro = np.dot(lpar_c,Pts)
        psi_a = Pts[:,np.argmax(pro)] # v_t_A (Line 4)
        psi_f = Pts[:,np.argmin(pro)] # v_t_FW-S (Line 5)
        
        if sparse:
            diff = (psi_a - psi_f).T.dot(lpar_c) 
        else:
            diff = np.dot(lpar_c, np.subtract(psi_a,psi_f))

        if (diff >= m._gap): # Line 6
            ### Simplex Gradient Descent ###
            sigd_count += 1
            d = pro - np.sum(pro)/Pts.shape[1] # line 3, Projection onto the hyperplane of the probability simplex}
            Pts_dot_d = Pts.dot(d)

            if (np.equal(d,0).all()): # line 4
                as_size = Pts.shape[1]
                psi_q = Pts[:,0]
                Pts = Pts[:,0]
                Vts = Vts[:,0]
                lamb = np.array([[1]])
                #(Pts, Vts, lamb) = prune(Pts, Vts, psi_q)
                as_drops += as_size - Pts.shape[1]

                # restart Nesterov accelerated SiGD
                psi_last = psi_q # y_1 in NAG
                lam_s = (1 + sqrt(1 + 4 * (lam_0 ** 2))) / 2
                lam_s_plus_1 = (1 + sqrt(1 + 4 * (lam_s ** 2))) / 2
                gam_s = (1 - lam_s) / lam_s_plus_1
            else:
                eta = np.divide(lamb,d[:,None]) # line 7
                eta = np.min(eta[d > 0]) # line 7

                # Equivalent to psi_n = Pts @ (lamb - eta*d)
                psi_n = psi_q - eta*(Pts_dot_d) # line 8, psi_n is y
                #psi_n = Pts @ (lamb.flatten() - eta*d)
                res = Y - lpar*psi_n[Xn]
                fn = np.dot(res,res)/n # fn is f(y)

                if (objVal >= fn): # line 9
                    if nag: # NAG update
                        psi_q = (1 - gam_s) * psi_n + gam_s * psi_last
                        psi_last = psi_n
                        res = Y - lpar*psi_q[Xn]
                        objVal = np.dot(res,res)/n # fn is f(y)

                        as_size = Pts.shape[1]
                        lamb = (1 - gam_s) * (lamb - eta*d[:,None]) + gam_s * lamb         
                    else:
                        psi_q = psi_n # line 10
                        objVal = fn
                        as_size = Pts.shape[1]
                        lamb = lamb - eta*d[:,None]
                        
                    inds = lamb.flatten() > 0
                    Pts = Pts[:, inds]
                    Vts = Vts[:, inds]
                    lamb = lamb[inds]/np.sum(lamb[inds])
                    #(Pts, Vts, lamb) = prune(Pts, Vts, psi_q)
                    as_drops += as_size - Pts.shape[1]
                else:
                    grap = Pts_dot_d
                    grap_Xn = grap[Xn]
                    gam = -np.dot(Y/lpar-psi_q[Xn], grap_Xn) / np.dot(grap_Xn, grap_Xn)
                    psi_n = psi_q - gam*grap
                    if nag: # NAG update
                        psi_q = (1 - gam_s) * psi_n + gam_s * psi_last
                        psi_last = psi_n
                        lamb = (1 - gam_s) * (lamb - gam*d[:,None]) + gam_s * lamb
                    else:
                        psi_q = psi_q - gam*grap
                        lamb = lamb - gam*d[:,None]
                    lamb = lamb/np.sum(lamb)
                
            if nag:# Update NAG parameters
                lam_s = (1 + sqrt(1 + 4 * (lam_s ** 2))) / 2
                lam_s_plus_1 = (1 + sqrt(1 + 4 * (lam_s ** 2))) / 2
                gam_s = (1 - lam_s) / lam_s_plus_1
                    
        else:        
            ### Weak Separation ###
            orcl_count += 1
            m._cmin = np.dot(lpar_c,psi_q) # <grad(f(x_0)), x_0>
            if (iter_count == 1):
                # solve linearized (integer) optimization problem
                ip_count += 1
                m.setObjective(lpar_c @ psi) # minimization
                m._oracle = "FullIP"
                m.optimize() 
                psi_n = psi.X
                the_n = the.X
                m._gap = (m._cmin - m.objVal)/2
            else:
                oflg = True # True if we do not find an "improving point"

                altmin_count = 0
                best_cmin = float('inf')
                ### Heuristic: Alternating Minimization ###
                while (oflg and altmin_count < 100):
                    altmin_count += 1
                    if (altmin_count == 1):
                        the_n = the.X
                    elif (altmin_count == 2):
                        the_n = 1-the.X
                    else:
                        the_n = np.round(np.random.uniform(0,1,np.sum(r)))

                    (psi_n, the_n, last_cmin) = altmin(r, lpar, p, tol, m._cmin, m._gap, c, the_n, Un, indices, pattern, altmin_pattern)
                    
                    if (m._cmin - last_cmin > m._gap): # first case in the output of Weak Separation Oracle
                        m._oracle = "AltMin"
                        oflg = False
                    elif (last_cmin < best_cmin):
                        best_cmin = last_cmin
                        psi_b = psi_n
                        the_b = the_n                        

                # improve the gap estimate when certain conditions hold
                if oflg and m._cmin - best_cmin > (objVal - bestbd)/2:
                    m._gap = (objVal - bestbd)/2
                    psi_n = psi_b
                    the_n = the_b
                    m._oracle = "AltMin"
                    oflg = False
                
                if oflg:
                    #PASS BEST SOLUTION SO FAR TO MIP SOLVER WHEN NEEDED
                    ip_count += 1
                    psi.Start = psi_b
                    the.Start = the_b
                    m.setObjective(lpar_c @ psi)
                    m._oracle = "FullIP"
                    m.optimize(callback)
                    psi_n = psi.X
                    the_n = the.X
                    if (m._cmin - m.objVal < m._gap): # second case in the output of the weak separation oracle
                        m._gap = m._gap/2 # line 13
                        # MAYBE UPDATE GAP USING FULLIP SOLUTION?!?!?!

            Vts = np.hstack((Vts,the_n[:,None]))
            if sparse:
                psi_n = Vts_to_Pts_sparse(the_n.reshape(-1, 1), X_uni, r, cum_r)
                Pts = sp.hstack([Pts, psi_n])
                psi_n = psi_n.toarray().squeeze()
            else:
                Pts = np.hstack((Pts,psi_n[:,None]))
            lamb_q = np.vstack((lamb,0))
            lamb_n = np.vstack((np.zeros(lamb.shape),1))
            (psi_q, lamb, gam) = golden(Xn, Y, psi_q, psi_n, lamb_q, lamb_n, lpar) # find the optimal cvx combination of the current iterate and the new vertex

            # restart Nesterov accelerated SiGD
            if nag:
                psi_last = psi_q # y_1 in NAG
                lam_s = (1 + sqrt(1 + 4 * (lam_0 ** 2))) / 2
                lam_s_plus_1 = (1 + sqrt(1 + 4 * (lam_s ** 2))) / 2
                gam_s = (1 - lam_s) / lam_s_plus_1
        
        res = Y - lpar*psi_q[Xn]
        objVal = np.dot(res,res)/n
        bestbd = np.max([bestbd, objVal - 2*m._gap])
        #bestbd = np.max([bestbd, objVal - 2*m._gap, objVal - 8*imup*m._gap**2])
        as_size = Pts.shape[1]
        
        if (2*m._gap < tol or (objVal - bestbd) < tol):
        #if (2*m._gap < tol or (objVal - bestbd) < tol or 8*imup*m._gap**2 < tol):
            is_true = False

        if (verbose & (prnt_count % 20 == 0)):
            print("")
            print("   Active Sets   |           Objective Bounds            |         Work")
            print("  Size    Drops  |  Incumbent       BestBd       AddGap  |  SiGD  IntPrg   Time")
            print("")

        curr_time = time.process_time()
        curr_time = time.time()
        elapsed_time = curr_time - last_time
        if (verbose & (prnt_count % 20 == 0 or not is_true or prnt_time <= curr_time - 5 or 0.9*last_gap > m._gap)):
            prnt_time = curr_time
            prnt_count += 1
            last_gap = m._gap
            print(" %5i   %6i  %12.3e %12.3e %12.3e   %6s %6s %6is" % 
                (as_size, as_drops, objVal, bestbd, objVal - bestbd, sigd_count, ip_count, elapsed_time))
            
    if verbose:
        print("\n")
        print("Optimal solution found (tolerance %7.2e)" % (tol))
        print("Best objective %10.6e, best bound %10.6e, additive gap %10.6e" % 
              (objVal, bestbd, objVal - bestbd))
    
    # recover the solution which is a cvx comb of 
    psi_q = np.zeros(np.prod(r))
    for ind in range(as_size):
        the_n = Vts[cum_r[0]:cum_r[1],ind]
        for k in range(1,p):
            the_n = np.tensordot(the_n, Vts[cum_r[k]:cum_r[k+1],ind], axes = 0)

        psi_q += lamb[ind]*the_n.flatten()
        
    if wideX:
        X = Xo
        
    return (lpar*psi_q)

def predict(psi_q, X, r):

    # Compute derived parameters
    p = len(r)
    
    if (X.shape[1] == p):
        # Convert X into flat indices
        return(psi_q[np.ravel_multi_index((X.T-1).tolist(), r)])
    else:
        # X already has flat indices
        return(psi_q[X])
