# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import functools as ft

import numpy as np

from pyfr.nputil import npeval, fuzzysort
from pyfr.util import memoize


class BaseElements(object):
    __metaclass__ = ABCMeta

    # Map from dimension number to list of dynamical variables
    _dynvarmap = None

    # Additional storage requirements
    _need_vect_fpts = False

    def __init__(self, basiscls, eles, cfg):
        self._be = None

        self._eles = eles
        self._cfg = cfg

        self.nspts = nspts = eles.shape[0]
        self.neles = neles = eles.shape[1]
        self.ndims = ndims = eles.shape[2]

        # Kernels we provide
        self.kernels = {}

        # Check the dimensionality of the problem
        if ndims != basiscls.ndims or ndims not in self._dynvarmap:
            raise ValueError('Invalid element matrix dimensions')

        # Determine the number of dynamical variables
        self.nvars = len(self._dynvarmap[ndims])

        # Instantiate the basis class
        self._basis = basis = basiscls(nspts, cfg)

        # Sizes
        self.nupts = basis.nupts
        self.nfpts = basis.nfpts
        self.nfacefpts = basis.nfacefpts

        # Physical normals at the flux points
        self._gen_pnorm_fpts()

        # Construct the physical location operator matrix
        plocop = basis.sbasis.nodal_basis_at(basis.fpts)

        # Apply the operator to the mesh elements and reshape
        plocfpts = np.dot(plocop, eles.reshape(nspts, -1))
        plocfpts = plocfpts.reshape(self.nfpts, neles, ndims)
        plocfpts = plocfpts.transpose(1, 2, 0).tolist()

        self._srtd_face_fpts = [[fuzzysort(pts, ffpts) for pts in plocfpts]
                                for ffpts in basis.facefpts]

    @abstractmethod
    def _process_ics(self, ics):
        pass

    def set_ics_from_cfg(self):
        # Bring simulation constants into scope
        vars = self._cfg.items_as('constants', float)

        if any(d in vars for d in 'xyz'):
            raise ValueError('Invalid constants (x, y, or z) in config file')

        # Construct the physical location operator matrix
        plocop = self._basis.sbasis.nodal_basis_at(self._basis.upts)

        # Apply the operator to the mesh elements and reshape
        plocupts = np.dot(plocop, self._eles.reshape(self.nspts, -1))
        plocupts = plocupts.reshape(self.nupts, self.neles, self.ndims)

        # Extract the components of the mesh coordinates
        coords = np.rollaxis(plocupts, 2)
        vars.update(dict(zip('xyz', coords)))

        # Evaluate the ICs from the config file
        ics = [npeval(self._cfg.get('soln-ics', dv), vars)
               for dv in self._dynvarmap[self.ndims]]

        # Allocate
        self._scal_upts = np.empty((self.nupts, self.nvars, self.neles))

        # Convert from primitive to conservative form
        for i, v in enumerate(self._process_ics(ics)):
            self._scal_upts[:,i,:] = v

    def set_ics_from_soln(self, solnmat, solncfg):
        # Recreate the existing solution basis
        currb = self._basis
        solnb = currb.__class__(None, solncfg)

        # Form the interpolation operator
        interp = solnb.ubasis.nodal_basis_at(currb.upts)

        # Sizes
        nupts, neles, nvars = self.nupts, self.neles, self.nvars

        # Apply and reshape
        self._scal_upts = np.dot(interp, solnmat.reshape(solnb.nupts, -1))
        self._scal_upts = self._scal_upts.reshape(nupts, nvars, neles)

    @abstractmethod
    def set_backend(self, backend, nscal_upts):
        # Ensure a backend has not already been set
        assert self._be is None
        self._be = backend

        # Sizes
        nupts, nfpts = self.nupts, self.nfpts
        nvars, ndims = self.nvars, self.ndims
        neles = self.neles

        # Convenience wrapper for aligned allocations
        matrix = ft.partial(backend.matrix, tags={'align'})

        # Allocate and bank the storage required by the time integrator
        self._scal_upts = [matrix((nupts, nvars, neles), self._scal_upts)
                           for i in xrange(nscal_upts)]
        self.scal_upts_inb = backend.matrix_bank(self._scal_upts)
        self.scal_upts_outb = backend.matrix_bank(self._scal_upts)

        # Allocate scratch space required for evaluating -∇·f
        self._vect_upts = matrix((ndims, nupts, nvars, neles))
        self._scal_fpts = matrix((nfpts, nvars, neles), extent='scal_fpts')

        if self._need_vect_fpts:
            self._vect_fpts = matrix((ndims, nfpts, nvars, neles),
                                     extent='vect_fpts')

    @memoize
    def opmat(self, name):
        return self._be.const_matrix(getattr(self._basis, name.lower()),
                                     tags={name})

    @memoize
    def smat_at(self, name):
        smat = self._get_smats(getattr(self._basis, name))
        return self._be.const_matrix(smat, tags={'align'})

    @memoize
    def rcpdjac_at(self, name):
        _, djac = self._get_smats(getattr(self._basis, name), retdets=True)

        if np.any(djac < -1e-5):
            raise RuntimeError('Negative mesh Jacobians detected')

        return self._be.const_matrix(1.0 / djac, tags={'align'})

    def _gen_pnorm_fpts(self):
        smats = self._get_smats(self._basis.fpts).transpose(1, 3, 0, 2)

        # We need to compute |J|*[(J^{-1})^{T}.N] where J is the
        # Jacobian and N is the normal for each fpt.  Using
        # J^{-1} = S/|J| where S are the smats, we have S^{T}.N.
        pnorm_fpts = np.einsum('ijlk,il->ijk', smats, self._basis.norm_fpts)

        # Compute the magnitudes of these flux point normals
        mag_pnorm_fpts = np.einsum('...i,...i', pnorm_fpts, pnorm_fpts)
        mag_pnorm_fpts = np.sqrt(mag_pnorm_fpts)

        # Normalize the physical normals at the flux points
        self._norm_pnorm_fpts = pnorm_fpts / mag_pnorm_fpts[...,None]
        self._mag_pnorm_fpts = mag_pnorm_fpts

    def _get_jac_eles_at(self, pts):
        nspts, neles, ndims = self.nspts, self.neles, self.ndims
        npts = len(pts)

        # Form the Jacobian operator
        jacop = np.rollaxis(self._basis.sbasis.jac_nodal_basis_at(pts), 2)

        # Cast as a matrix multiply and apply to eles
        jac = np.dot(jacop.reshape(-1, nspts), self._eles.reshape(nspts, -1))

        # Reshape (npts*ndims, neles*ndims) => (npts, ndims, neles, ndims)
        jac = jac.reshape(npts, ndims, neles, ndims)

        # Transpose to get (ndims, npts, ndims, neles)
        return jac.transpose(3, 0, 1, 2)

    def _get_smats(self, pts, retdets=False):
        jac = self._get_jac_eles_at(pts)
        smats = np.empty_like(jac)

        if self.ndims == 2:
            a, b, c, d = jac[0,:,0], jac[0,:,1], jac[1,:,0], jac[1,:,1]

            smats[0,:,0], smats[0,:,1] =  d, -b
            smats[1,:,0], smats[1,:,1] = -c,  a

            if retdets:
                djacs = a*d - b*c
        else:
            # We note that J = [x0, x1, x2]
            x0, x1, x2 = jac[:,:,0], jac[:,:,1], jac[:,:,2]

            smats[0] = np.cross(x1, x2, axisa=0, axisb=0, axisc=1)
            smats[1] = np.cross(x2, x0, axisa=0, axisb=0, axisc=1)
            smats[2] = np.cross(x0, x1, axisa=0, axisb=0, axisc=1)

            if retdets:
                # Exploit the fact that det(J) = x0 . (x1 ^ x2)
                djacs = np.einsum('ij...,ji...->j...', x0, smats[0])

        return (smats, djacs) if retdets else smats

    def get_mag_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[fidx][eidx]
        return self._mag_pnorm_fpts[fpts_idx,eidx]

    def get_norm_pnorms_for_inter(self, eidx, fidx):
        fpts_idx = self._srtd_face_fpts[fidx][eidx]
        return self._norm_pnorm_fpts[fpts_idx,eidx]

    def get_scal_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        cstri = [(self._scal_fpts.leadsubdim,)]*nfp

        return [self._scal_fpts]*nfp, rcmap, cstri

    def get_vect_fpts_for_inter(self, eidx, fidx):
        nfp = self.nfacefpts[fidx]

        rcmap = [(fpidx, eidx) for fpidx in self._srtd_face_fpts[fidx][eidx]]
        rcstri = [(self.nfpts, self._vect_fpts.leadsubdim)]*nfp

        return [self._vect_fpts]*nfp, rcmap, rcstri
