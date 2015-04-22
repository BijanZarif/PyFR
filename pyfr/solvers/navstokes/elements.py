# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base.kernels import ComputeMetaKernel
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


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

        if self.cfg.getfloat('solver-avis', 'amu0', 0.0):
            amu = self._avis_upts
            tplargs.update(dict(art_vis='mu'))
        else:
            amu = None
            tplargs.update(dict(art_vis='none'))

        if 'flux' in self.antialias:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts, amu=amu
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts, amu=amu
            )

        # Artificial Viscosity
        if amu:
            backend.pointwise.register('pyfr.solvers.navstokes.kernels.entropy')
            backend.pointwise.register('pyfr.solvers.navstokes.kernels.avis')

            def artf_vis():
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
                tplargs.update(dict(nupts=self.nupts, nfpts=self.nfpts,
                                    order=self._basis.order, ubdegs=ubdegs))

                # Column view for avis_upts/fpts
                def col_view(mat):
                    ioshape = mat.ioshape
                    dim = len(ioshape)
                    if dim == 3:
                        vshape = (ioshape[0],)
                        nelespts = ioshape[-1]
                        rcmap = np.array([[j, i] for i in range(ioshape[2]) for j in range(ioshape[1])])
                        matmap = np.array([mat.mid]*nelespts)
                        stridemap = np.array([[mat.leaddim]]*nelespts)

                    return backend.view(matmap, rcmap, stridemap, vshape)

                self._avis_upts_cv = col_view(self._avis_upts)

                self._avis_fpts_cv = col_view(self._avis_fpts)
                self._avis_upts_temp_cv = col_view(self._avis_upts_temp)
                avis = backend.kernel('avis', tplargs, dims=[self.neles],
                                      ubdegs=ubdegs,
                                      s=self._avis_upts_temp_cv,
                                      amu_e=self._avis_upts_cv, amu_f=self._avis_fpts_cv)

                return ComputeMetaKernel([ent, mul, avis])

            self.kernels['avis'] = artf_vis
