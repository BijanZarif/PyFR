# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)
from pyfr.solvers.baseadvec.elements import get_mv_grid_terms


class EulerIntInters(BaseAdvectionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.intcflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        # Update moving grid velocity terms
        mvex, mode, plocfpt = get_mv_grid_terms(self, self.cfg, self._privarmap, args[1])
        tplargs.update(mvex)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
             ul=self._scal0_lhs, ur=self._scal0_rhs,
             magnl=self._mag_pnorm_lhs, magnr=self._mag_pnorm_rhs,
             nl=self._norm_pnorm_lhs, ploc=plocfpt
        )

        if self.cfg.get('solver-moving-terms', 'mode', None) == 'rotation':
            self._rotfvec('pnorm_rot', self._norm_pnorm_lhs)
            self._rotfvec('plocfpts_rot', plocfpt)


class EulerMPIInters(BaseAdvectionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.mpicflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        # Update moving grid velocity terms
        mode = None
        mvex, mode, plocfpt = get_mv_grid_terms(self, self.cfg, self._privarmap, args[1])
        tplargs.update(mvex)

        if self.cfg.get('solver-moving-terms', 'mode', None) == 'rotation':
            self._rotfvec('pnorm_rot', self._norm_pnorm_lhs)
            self._rotfvec('plocfpts_rot', plocfpt)

        if mode is None:
            self.kernels['comm_flux'] = lambda: self._be.kernel(
                'mpicflux', tplargs, dims=[self.ninterfpts],
                 ul=self._scal0_lhs, ur=self._scal0_rhs,
                 magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs, ploc=plocfpt
            )
        else:
            import functools as ft
            import numpy as np
            import re
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
            mscal_tmpm = self._be.matrix((nfpts, self.nvars, self.ninters), tags={'align'})

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

            matmap = np.array([mscal_tmpm.mid]*self.ninterfpts)
            stridemap = np.array([[mscal_tmpm.leadsubdim]]*self.ninterfpts)
            mscal_tmpv = self._be.view(matmap, rcmap, stridemap, vshape)

            # Original norm and mag for pnorm and ploc_fpts
            norm_pnorm_int = self._norm_pnorm_lhs
            mag_pnorm_int = self._mag_pnorm_lhs

            # Additional norm and plocfpts
            norm_pnorm_lhs0 = self._be.matrix(norm_pnorm_int.ioshape, initval= norm_pnorm_int.get(), tags={'align'})
            norm_pnorm_lhs1 = self._be.matrix(norm_pnorm_int.ioshape, initval= norm_pnorm_int.get(), tags={'align'})

            if mode == 'rotation':
                plocfpt_lhs0 = self._be.matrix(plocfpt.ioshape, tags={'align'})
                plocfpt_lhs1 = self._be.matrix(plocfpt.ioshape, tags={'align'})
            else:
                plocfpt_lhs0 = None
                plocfpt_lhs1 = None

            # Register copy and flux kernel
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.mpicfluxs')
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.viewcopy')
            self._be.pointwise.register('pyfr.solvers.euler.kernels.mortar.mpiviewcopy')

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

            ismv = tplargs['ismv']

            if mode == 'translation':
                # Translation
                vs = np.array([eval(v) for v in tplargs['mvex']])
                # Slide velocity should be aligned with slide plane
                vs_mag = np.linalg.norm(vs)
                dist = np.linalg.norm(efpts[0][1] - efpts[0][0])
                sign = np.dot(vs, efpts[0][1] - efpts[0][0])
                peri = len(efpts)*dist

            elif mode == 'rotation':
                # Rotation
                vs_mag = cfg.get('solver-moving-terms', 'rot-vel', '0')
                vs_mag = eval(re.sub(r'\b(pi)\b', 'np.pi', vs_mag))
                peri = 2.0*np.pi
                dist = peri/len(efpts)
                p2 = efpts[0][1]
                p1 = efpts[0][0]
                sign = np.arctan2(p2[1], p2[0]) - np.arctan2(p1[1], p1[0])

            if sign < 0: vs_mag = -vs_mag
            if ismv > 0.0: vs_mag = -vs_mag

            # Prepare Kernel (Python)
            def prepare_mortar():
                class prepare(ComputeKernel):
                    def run(self, queue, t=0):
                        # Relative position (considering the uniform slide plane)
                        move = vs_mag*t
                        move -= peri*int(move/peri)

                        # Move normalized by one edge
                        n = move/dist

                        # Movement on edge coordinate [-1,0] or [1,0]
                        res = n - int(n)

                        # How many edges to be passed
                        n = int(n)

                        if vs_mag > 0.0:
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

                        if mode == 'rotation':
                            # Pnorm interpolation
                            # off, len = 0, res
                            # R = fb.nodal_basis_at(2*off-1 + len*(pts + 1))
                            R = np.dot(invM, S0)
                            tmp = np.einsum('ij, klj->kli', R, norm_pnorm_int.get().reshape(2, -1, nfpts))
                            norm_pnorm_lhs0.set(tmp.reshape(2, -1))

                            tmp = np.einsum('ij, klj->kli', R, plocfpt.get().reshape(2, -1, nfpts))
                            plocfpt_lhs0.set(tmp.reshape(2, -1))

                            # off, len = res, 1.0-res
                            # R = fb.nodal_basis_at(2*off-1 + len*(pts + 1))
                            R = np.dot(invM, S1)
                            tmp = np.einsum('ij, klj->kli', R, norm_pnorm_int.get().reshape(2, -1, nfpts))
                            norm_pnorm_lhs1.set(tmp.reshape(2, -1))

                            tmp = np.einsum('ij, klj->kli', R, plocfpt.get().reshape(2, -1, nfpts))
                            plocfpt_lhs1.set(tmp.reshape(2, -1))
                return prepare()

            def slide_flux():
                copy_pre = [self._be.kernel('viewcopy', tplargs=tplargs,
                                          dims=[self.ninterfpts], ul=self._scal0_lhs, ur=mscal_tmpv),
                            # Multiplication
                            self._be.kernel('mul', P0, mscal_tmpm, out=mscal_fpts0),
                            self._be.kernel('mul', P1, mscal_tmpm, out=mscal_fpts1),

                            self._be.kernel('mpiviewcopy', tplargs=tplargs,
                                          dims=[self.ninterfpts], ul=self._scal0_rhs, ur=mscal_tmpv),
                            # Multiplication
                            self._be.kernel('mul', P2, mscal_tmpm, out=mscal_fpts2),
                            self._be.kernel('mul', P3, mscal_tmpm, out=mscal_fpts3),
                            ]

                comm = [self._be.kernel('mpicfluxs', tplargs, dims=[self.ninterfpts],
                                       ul=mscal_lhs0, ur=mscal_rhs1,
                                       magnl=mag_pnorm_int, nl=norm_pnorm_lhs0, ploc=plocfpt_lhs0),
                        self._be.kernel('mpicfluxs', tplargs, dims=[self.ninterfpts],
                                       ul=mscal_lhs1, ur=mscal_rhs0,
                                       magnl=mag_pnorm_int, nl=norm_pnorm_lhs1, ploc=plocfpt_lhs1),
                        ]

                copy_post = [self._be.kernel('mul', invP0, mscal_fpts0, out=mscal_tmpm),
                             self._be.kernel('mul', invP1, mscal_fpts1, out=mscal_tmpm, beta=1.0),
                             self._be.kernel('viewcopy', tplargs=tplargs, dims=[self.ninterfpts],
                                             ur=self._scal0_lhs, ul=mscal_tmpv)
                             ]

                return ComputeMetaKernel(copy_pre + comm + copy_post)

            self.kernels['premortar'] = prepare_mortar
            self.kernels['comm_flux'] = slide_flux


class EulerBaseBCInters(BaseAdvectionBCInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.euler.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type)

        # Update moving grid velocity terms
        mvex, mode, plocfpt = get_mv_grid_terms(self, self.cfg, self._privarmap, args[1])
        tplargs.update(mvex)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs, dims=[self.ninterfpts], ul=self._scal0_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs, ploc=plocfpt
        )

        if self.cfg.get('solver-moving-terms', 'mode', None) == 'rotation':
            self._rotfvec('pnorm_rot', self._norm_pnorm_lhs)
            self._rotfvec('plocfpts_rot', plocfpt)


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
