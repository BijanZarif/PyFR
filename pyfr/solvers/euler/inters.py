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

        if mode is not None:
            self.kernels['scal_fpts_unpack'] = lambda: self._be.kernel(
                'unpack_slide', self._scal0_rhs, mode, tplargs['mvex'], tplargs['ismv'], self.endfpts_at(args[1])
            )

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs, dims=[self.ninterfpts],
             ul=self._scal0_lhs, ur=self._scal0_rhs,
             magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
             ploc=plocfpt
        )

        if self.cfg.get('solver-moving-terms', 'mode', None) == 'rotation':
            self._rotfvec('pnorm_rot', self._norm_pnorm_lhs)
            self._rotfvec('plocfpts_rot', plocfpt)


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
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=plocfpt
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


class EulerSupOutflowBCInters(EulerBaseBCInters):
    type = 'sup-out-fn'
