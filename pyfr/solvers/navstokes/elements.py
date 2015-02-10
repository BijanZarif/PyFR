# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements
from pyfr.solvers.baseadvec.elements import get_mv_grid_terms


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    def set_backend(self, backend, nscalupts):
        super().set_backend(backend, nscalupts)
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')

        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr not in {'sutherland', 'none'}:
            raise ValueError('Invalid viscosity-correction option')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       visc_corr=visc_corr,
                       c=self.cfg.items_as('constants', float))

        if 'flux' in self.antialias:
            # Update moving grid velocity terms
            mvex, mode, plocqpts = get_mv_grid_terms(self, self.cfg, self._privarmap, 'qpts')
            tplargs.update(mvex)

            self._smats_u = self.smat_at('qpts')

            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self._smats_u,
                f=self._vect_qpts,
                ploc=plocqpts
            )

            if self.cfg.get('solver-moving-terms', 'mode', None) == 'rotation':
                backend.pointwise.register('pyfr.solvers.baseadvec.kernels.rotfvec')
                self.kernels['plocupts_rot'] = lambda: backend.kernel(
                    'rotfvec', tplargs=tplargs, dims=[self.nupts, self.neles],
                    vecs=plocqpts
                )

        else:
            # Update moving grid velocity terms
            mvex, mode, plocupts = get_mv_grid_terms(self, self.cfg, self._privarmap, 'upts')
            tplargs.update(mvex)

            self._smats_u = self.smat_at('upts')

            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self._smats_u,
                f=self._vect_upts,
                ploc=plocupts
            )

            if self.cfg.get('solver-moving-terms', 'mode', None) == 'rotation':
                backend.pointwise.register('pyfr.solvers.baseadvec.kernels.rotfvec')
                self.kernels['plocupts_rot'] = lambda: backend.kernel(
                    'rotfvec', tplargs=tplargs, dims=[self.nupts, self.neles],
                    vecs=plocupts
                )

        if self.cfg.get('solver-moving-terms', 'mode', None) == 'rotation':
            backend.pointwise.register('pyfr.solvers.baseadvec.kernels.rotsmat')
            self.kernels['smats_rot'] = lambda: backend.kernel(
                'rotsmat', tplargs=tplargs, dims=[self.nupts, self.neles],
                smats=self._smats_u
            )