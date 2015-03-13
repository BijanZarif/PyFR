# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class EulerIntInters(BaseAdvectionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.intcflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
             ul=self._scal0_lhs, ur=self._scal0_rhs,
             magnl=self._mag_pnorm_lhs, magnr=self._mag_pnorm_rhs,
             nl=self._norm_pnorm_lhs
        )


class EulerMPIInters(BaseAdvectionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.mpicflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs, dims=[self.ninterfpts],
             ul=self._scal0_lhs, ur=self._scal0_rhs,
             magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )

        if True:
            import functools as ft
            import numpy as np
            from pyfr.backends.base.kernels import ComputeKernel, ComputeMetaKernel
            from pyfr.nputil import chop
            from pyfr.quadrules import get_quadrule
            from pyfr.shapes import get_polybasis

            print('Sliding Mesh Flux')
            cfg = self.cfg

            # Make additional matrix and view
            ninters = self.ninters
            ninterfpts = self.ninterfpts
            be = self._be
            nfpts = int(self.ninterfpts/self.ninters)
            mscal_fpts0 = self._be.matrix((nfpts, self.nvars, self.ninters), tags={'align'})
            mscal_fpts1 = self._be.matrix((nfpts, self.nvars, self.ninters), tags={'align'})
            mscal_fpts2 = self._be.matrix((nfpts, self.nvars, self.ninters), tags={'align'})
            mscal_fpts3 = self._be.matrix((nfpts, self.nvars, self.ninters), tags={'align'})
            mscal_mtmp = self._be.matrix((nfpts, self.nvars, self.ninters), tags={'align'})

            vshape = (self.nvars,)
            rcmap = np.array([[j, i] for i in range(self.ninters) for j in range(nfpts)])

            matmap = np.array([mscal_fpts0.mid]*self.ninterfpts)
            stridemap = np.array([[mscal_fpts0.leadsubdim]]*self.ninterfpts)
            mscal_lhs0 = self._be.view(matmap, rcmap, stridemap, vshape)

            matmap = np.array([mscal_fpts1.mid]*self.ninterfpts)
            stridemap = np.array([[mscal_fpts1.leadsubdim]]*self.ninterfpts)
            mscal_lhs1 = self._be.view(matmap, rcmap, stridemap, vshape)

            matmap = np.array([mscal_fpts2.mid]*self.ninterfpts)
            stridemap = np.array([[mscal_fpts2.leadsubdim]]*self.ninterfpts)
            mscal_rhs0 = self._be.view(matmap, rcmap, stridemap, vshape)

            matmap = np.array([mscal_fpts3.mid]*self.ninterfpts)
            stridemap = np.array([[mscal_fpts3.leadsubdim]]*self.ninterfpts)
            mscal_rhs1 = self._be.view(matmap, rcmap, stridemap, vshape)

            matmap = np.array([mscal_mtmp.mid]*self.ninterfpts)
            stridemap = np.array([[mscal_mtmp.leadsubdim]]*self.ninterfpts)
            mscal_vtmp = self._be.view(matmap, rcmap, stridemap, vshape)

            # Register copy and flux kernel
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.mpicfluxs')
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.viewcopy')
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.viewtomat')
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.mattoview')
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.mpiviewcopy')
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.viewmpicopy')

            # Pre-defined (for only line)
            kind = 'line'
            quad_map = ('gauss-legendre', 8)
            qr = get_quadrule(kind, *quad_map)
            rule = cfg.get('solver-interfaces-' + kind, 'flux-pts')
            order = cfg.getint('solver', 'order')

            # Surf-flux check
            antialias = cfg.get('solver', 'anti-alias', 'none')
            antialias = {s.strip() for s in antialias.split(',')}
            if 'surf-flux' in antialias:
                qdeg = cfg.getint('solver-interfaces-' + kind, 'quad-deg')
                pts = get_quadrule(kind, rule=rule, qdeg=qdeg).pts
            else:
                pts = get_quadrule(kind, rule=rule, npts=order+1).pts

            fb = get_polybasis(kind, order+1, pts)

            # Matrix function
            @ft.lru_cache(maxsize=None)
            @chop
            def projmat(off, len):
                fe = fb.nodal_basis_at(qr.pts)
                fp = fb.nodal_basis_at(2*off-1 + len*(qr.pts + 1))

                return np.einsum('i...,ij,ik', qr.wts, fe, fp)

            M = projmat(0, 1)
            invM = np.linalg.inv(M)

            # Operation Matrix
            P0 = self._be.const_matrix(invM, tags={'align'})
            P1 = self._be.const_matrix(invM, tags={'align'})
            P2 = self._be.const_matrix(invM, tags={'align'})
            P3 = self._be.const_matrix(invM, tags={'align'})

            invP0 = self._be.const_matrix(invM, tags={'align'})
            invP1 = self._be.const_matrix(invM, tags={'align'})

            # Relative difference
            lhs = args[1]
            efpts = self.endfpts_at(lhs)

            # Temporary
            vs_mag = 0.2
            dist = np.linalg.norm(efpts[0][1] - efpts[0][0])
            sign = 1.0
            peri = len(efpts)*dist

            from pyfr.mpiutil import get_comm_rank_root
            comm, rank, root = get_comm_rank_root()

            ismv = 0.0
            if rank == 0:
                ismv = 1.0

            if sign < 0:
                vs_mag = -vs_mag
            if ismv > 0.0:
                vs_mag = -vs_mag

            def slide_flux():
                # Prepare Kernel (Python)
                class prepare(ComputeKernel):
                    def run(self, queue):
                        # Relative position (considering the uniform slide plane)
                        t = 1.0
                        move = vs_mag*t
                        move -= peri*int(move/peri)

                        # Move normalized by one edge
                        n = move/dist

                        # Movement on edge coordinate [-1,0] or [1,0]
                        res = n - int(n)

                        # How many edges to be passed
                        n = int(n)

                        if res >= -1e-15:
                            idx1 = list(range(ninters - n, ninters)) + list(range(0, ninters - n))
                            idx2 = list(range(ninters - 1 - n, ninters)) + list(range(0, ninters - 1 - n))
                        else:
                            res += 1.0
                            n -= 1

                            idx1 = list(range(-n, ninters)) + list(range(0, - n))
                            idx2 = list(range(-n - 1, ninters)) + list(range(0, - n - 1))

                        # Update View
                        rcmap = np.array([[j, i] for i in idx1 for j in range(nfpts)])
                        matmap = np.array([mscal_fpts2.mid]*ninterfpts)
                        mscal_rhs0.update(matmap, rcmap)

                        rcmap = np.array([[j, i] for i in idx2 for j in range(nfpts)])
                        matmap = np.array([mscal_fpts3.mid]*ninterfpts)
                        mscal_rhs1.update(matmap, rcmap)

                        # Projection Matrix
                        S0 = projmat(0, res)
                        S1 = projmat(res,  1.0-res)
                        S2 = projmat(0, 1.0-res)
                        S3 = projmat(1.0-res, res)

                        P0._set(np.dot(invM, S0))
                        P1._set(np.dot(invM, S1))
                        P2._set(np.dot(invM, S2))
                        P3._set(np.dot(invM, S3))

                        invP0._set(res*np.dot(invM, S0.transpose()))
                        invP1._set((1.0-res)*np.dot(invM, S1.transpose()))
                        pass

                copy_pre = [self._be.kernel('viewcopy', tplargs=tplargs,
                                          dims=[self.ninterfpts], ul=self._scal0_lhs, ur=mscal_vtmp),
                            # self._be.kernel('copy', mscal_fpts0, mscal_mtmp),
                            # self._be.kernel('copy', mscal_fpts1, mscal_mtmp),
                            # Multiplication
                            self._be.kernel('mul', P0, mscal_mtmp, out=mscal_fpts0),
                            self._be.kernel('mul', P1, mscal_mtmp, out=mscal_fpts1),

                            self._be.kernel('mpiviewcopy', tplargs=tplargs,
                                          dims=[self.ninterfpts], ul=self._scal0_rhs, ur=mscal_vtmp),
                            # self._be.kernel('copy', mscal_fpts2, mscal_mtmp),
                            # self._be.kernel('copy', mscal_fpts3, mscal_mtmp)
                            # Multiplication
                            self._be.kernel('mul', P2, mscal_mtmp, out=mscal_fpts2),
                            self._be.kernel('mul', P3, mscal_mtmp, out=mscal_fpts3),
                            ]

                comm = [self._be.kernel('mpicfluxs', tplargs, dims=[self.ninterfpts],
                                       ul=mscal_lhs0, ur=mscal_rhs1,
                                       magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs),
                        self._be.kernel('mpicfluxs', tplargs, dims=[self.ninterfpts],
                                       ul=mscal_lhs1, ur=mscal_rhs0,
                                       magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs),
                        ]

                copy_post = [# self._be.kernel('viewcopy', tplargs=tplargs,
                             #             dims=[self.ninterfpts], ur=self._scal0_lhs, ul=mscal_lhs0),
                             self._be.kernel('mul', invP0, mscal_fpts0, out=mscal_mtmp),
                             self._be.kernel('mul', invP1, mscal_fpts1, out=mscal_mtmp, beta=1.0),
                             self._be.kernel('viewcopy', tplargs=tplargs, dims=[self.ninterfpts], ur=self._scal0_lhs, ul=mscal_vtmp)
                             # Sum / Mul
                             # self._be.kernel('viewmpicopy', tplargs=tplargs,
                             #             dims=[self.ninterfpts], ur=self._scal0_rhs, ul=mscal_rhs0)
                             ]

                return ComputeMetaKernel([prepare()] + copy_pre + comm + copy_post)

            self.kernels['comm_flux'] = slide_flux


class EulerBaseBCInters(BaseAdvectionBCInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal0_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class EulerSupInflowBCInters(EulerBaseBCInters):
    type = 'sup-in-fa'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['rho'], self._tpl_c['p'] = self._eval_opts(['rho', 'p'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims])


class EulerCharRiemInvBCInters(EulerBaseBCInters):
    type = 'char-riem-inv'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['p'], self._tpl_c['rho'] = self._eval_opts(['p', 'rho'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims])


class EulerSlpAdiaWallBCInters(EulerBaseBCInters):
    type = 'slp-adia-wall'
