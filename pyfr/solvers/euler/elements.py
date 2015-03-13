# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements
from pyfr.solvers.baseadvec.elements import get_mv_grid_terms


class BaseFluidElements(object):
    _privarmap = {2: ['rho', 'u', 'v', 'p'],
                  3: ['rho', 'u', 'v', 'w', 'p']}

    _convarmap = {2: ['rho', 'rhou', 'rhov', 'E'],
                  3: ['rho', 'rhou', 'rhov', 'rhow', 'E']}

    @staticmethod
    def pri_to_conv(pris, cfg):
        rho, p = pris[0], pris[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in pris[1:-1]]

        # Compute the energy
        gamma = cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in pris[1:-1])

        return [rho] + rhovs + [E]

    @staticmethod
    def conv_to_pri(convs, cfg):
        rho, E = convs[0], convs[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in convs[1:-1]]

        # Compute the pressure
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, backend, nscalupts):
        super().set_backend(backend, nscalupts)

        # Register our flux kernel
        backend.pointwise.register('pyfr.solvers.euler.kernels.tflux')

        # Template parameters for the flux kernel
        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
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
