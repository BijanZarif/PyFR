# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseInters, get_opt_view_perm
from pyfr.nputil import npeval


class BaseAdvectionIntInters(BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)

        const_mat = self._const_mat

        # Compute the `optimal' permutation for our interface
        self._gen_perm(lhs, rhs)

        # Generate the left and right hand side view matrices
        self._scal0_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._scal0_rhs = self._scal_view(rhs, 'get_scal_fpts_for_inter')

        # Generate the constant matrices
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._mag_pnorm_rhs = const_mat(rhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

    def _gen_perm(self, lhs, rhs):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter',
                                       self._elemap)


class BaseAdvectionMPIInters(BaseInters):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)
        self._rhsrank = rhsrank
        self._rallocs = rallocs

        const_mat = self._const_mat

        # Ordering for sliding mesh
        self._gen_perm(lhs)

        # Generate the left hand view matrix and its dual
        self._scal0_lhs = self._scal_xchg_view(lhs, 'get_scal_fpts_for_inter')
        self._scal0_rhs = be.xchg_matrix_for_view(self._scal0_lhs)

        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

        # Kernels
        self.kernels['scal_fpts_pack'] = lambda: be.kernel(
            'pack', self._scal0_lhs
        )
        self.kernels['scal_fpts_send'] = lambda: be.kernel(
            'send_pack', self._scal0_lhs, self._rhsrank, self.MPI_TAG
        )
        self.kernels['scal_fpts_recv'] = lambda: be.kernel(
            'recv_pack', self._scal0_rhs, self._rhsrank, self.MPI_TAG
        )
        self.kernels['scal_fpts_unpack'] = lambda: be.kernel(
            'unpack', self._scal0_rhs
        )

    def _gen_perm(self, lhs):
        fpts = self.endfpts_at(lhs)
        stride = self.ninterfpts // self.ninters
        self._perm = self._perm_line(fpts, stride)

    @staticmethod
    def _perm_line(fpts, stride):
        import numpy as np
        swa_fpts = fpts.swapaxes(0, 1)
        tmp = np.arange(len(swa_fpts[0]), dtype=np.int32)
        ninter = len(swa_fpts[0])

        end_pts = swa_fpts[1][0]
        igrp, egrp = 0, 1
        dir = 1
        for i in range(ninter - 2):
            i_curr = abs(tmp[i])

            # Search both side
            for j in range(ninter):
                if np.linalg.norm(swa_fpts[igrp][j] - end_pts) < 1e-8:
                    tmp[i+1] = dir*j
                    next_pts = swa_fpts[egrp][j]
                    break

                elif j != i_curr and np.linalg.norm(swa_fpts[egrp][j] - end_pts) < 1e-8:
                    dir *= -1
                    tmp[i+1] = dir*j
                    next_pts = swa_fpts[igrp][j]
                    igrp, egrp = egrp, igrp
                    break

            end_pts = next_pts

        # Last Point
        i += 1
        i_curr = 0
        end_pts = swa_fpts[0][0]

        # Search different side
        for j in range(ninter):
            if np.linalg.norm(swa_fpts[1][j] - end_pts) < 1e-8:
                tmp[i+1] = j

            if j != i_curr and np.linalg.norm(swa_fpts[0][j] - end_pts) < 1e-8:
                tmp[i+1] = -j

        # tmp = [0, 8, 17, 13, 5, -14, -6, -2, -11, -19, -10, -1, -15, -12, -3, 16, 7, 4, 18, 9]
        # tmp = [0, 6, 16,25, 37, 1, 21, 33, 4, 15, 7, 29, 9, 20, 30, 8, 3, 34, 18, -12, -38, -14, -23, -11, -32, -10, -39, -27, -26, -24, -35, -17, -2, -22, -5, -36, -13, -19, -31, 28]

        perm = []
        for t in tmp:
            if t > -1:
                perm.append(t*stride + np.arange(stride))
            else:
                perm.append((-t+1)*stride - 1 - np.arange(stride))

        perm = np.concatenate(perm)

        return perm



class BaseAdvectionBCInters(BaseInters):
    type = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfg)
        self.cfgsect = cfgsect

        const_mat = self._const_mat

        # For BC interfaces, which only have an LHS state, we take the
        # permutation which results in an optimal memory access pattern
        # iterating over this state.
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter', elemap)

        # LHS view and constant matrices
        self._scal0_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._mag_pnorm_lhs = const_mat(lhs, 'get_mag_pnorms_for_inter')
        self._norm_pnorm_lhs = const_mat(lhs, 'get_norm_pnorms_for_inter')

    def _eval_opts(self, opts, default=None):
        # Boundary conditions, much like initial conditions, can be
        # parameterized by values in [constants] so we must bring these
        # into scope when evaluating the boundary conditions
        cc = self.cfg.items_as('constants', float)

        cfg, sect = self.cfg, self.cfgsect

        # Evaluate any BC specific arguments from the config file
        if default is not None:
            return [npeval(cfg.get(sect, k, default), cc) for k in opts]
        else:
            return [npeval(cfg.get(sect, k), cc) for k in opts]
