# -*- coding: utf-8 -*-

from pyfr.backends.base.kernels import ComputeMetaKernel
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    def set_backend(self, backend, nscalupts):
        super(NavierStokesElements, self).set_backend(backend, nscalupts)
        backend.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')

        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr not in {'sutherland', 'none'}:
            raise ValueError('Invalid viscosity-correction option')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       visc_corr=visc_corr,
                       c=self.cfg.items_as('constants', float))

        if self.cfg.get('solver-avis', 'amu0', '0'):
            amu = self._avis_upts
            tplargs.update(dict(art_vis='mu'))
        else:
            amu = None
            tplargs.update(dict(art_vis='none'))

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),amu=amu,
                f=self._vect_qpts
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),amu=amu,
                f=self._vect_upts
            )

        # Artificial Viscosity
        if amu:
            backend.pointwise.register('pyfr.solvers.navstokes.kernels.entropy')
            backend.pointwise.register('pyfr.solvers.navstokes.kernels.avis')

            def artf_vis():
                import numpy as np
                # Compute Entropy and save to avis_upts
                ent = backend.kernel('entropy', tplargs=tplargs, dims=[self.nupts, self.neles],
                                      u=self.scal_upts_inb, s=self._avis_upts)

                # Compute modal of entropy and save to avis_upts_temp
                inVdm = np.linalg.inv(self._basis.ubasis.vdm.T)
                inVdm = self._be.const_matrix(inVdm, tags={'inVdm', 'align'})
                mul = backend.kernel('mul', inVdm, self._avis_upts, out=self._avis_upts_temp)

                # Element-wise Operation
                ubdegs = self._basis.ubasis.degrees
                tplargs['c'].update(self.cfg.items_as('solver-avis', float))
                tplargs.update(dict(nupts=self.nupts, nfpts=self.nfpts, lds=self._avis_upts.leaddim,
                                    order=self._basis.order, ubdegs=ubdegs))

                avis = backend.kernel('avis', tplargs, dims=[self.neles],
                                      ubdegs=ubdegs,
                                      s=self._avis_upts_temp,
                                      amu_e=self._avis_upts, amu_f=self._avis_fpts)

                return ComputeMetaKernel([ent, mul, avis])

            self.kernels['avis'] = artf_vis

