# -*- coding: utf-8 -*-

import pycuda.driver as cuda

from pyfr.backends.base import ComputeKernel
from pyfr.backends.base.packing import BasePackingKernels
from pyfr.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block


class CUDAPackingKernels(CUDAKernelProvider, BasePackingKernels):
    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, 'iiiPPPPP')

        # Compute the grid and thread-block size
        block = (128, 1, 1)
        grid = get_grid_for_block(block, v.n)

        # Create a CUDA event
        event = cuda.Event(cuda.event_flags.DISABLE_TIMING)

        class PackXchgViewKernel(ComputeKernel):
            def run(self, queue):
                scomp = queue.cuda_stream_comp
                scopy = queue.cuda_stream_copy

                # Pack
                kern.prepared_async_call(grid, block, scomp, v.n, v.nvrow,
                                         v.nvcol, v.basedata, v.mapping,
                                         v.cstrides or 0, v.rstrides or 0, m)

                # Copy the packed buffer to the host
                event.record(scomp)
                scopy.wait_for_event(event)
                cuda.memcpy_dtoh_async(m.hdata, m.data, scopy)

        return PackXchgViewKernel()

    def unpack(self, mv):
        class UnpackXchgMatrixKernel(ComputeKernel):
            def run(self, queue, t):
                cuda.memcpy_htod_async(mv.data, mv.hdata,
                                       queue.cuda_stream_comp)

        return UnpackXchgMatrixKernel()

    def unpack_slide(self, mv, mode, vs, ismv, fpts):
        from pyfr.quadrules import get_quadrule
        from pyfr.polys import get_polybasis
        import numpy as np
        import re

        ioshape = mv.ioshape

        cfg = self.backend.cfg
        order = cfg.getint('solver', 'order')

        # 2-D only
        kind = 'line'
        rule = cfg.get('solver-interfaces-' + kind, 'flux-pts')
        npts = order + 1
        pts = get_quadrule(kind, rule, npts).pts
        basis = get_polybasis(kind, order + 1, pts)

        if mode == 'translation':
            # Translation
            vs = np.array([eval(v) for v in vs])
            # Slide velocity should be aligned with slide plane
            vs_mag = np.linalg.norm(vs)
            dist = np.linalg.norm(fpts[0][1] - fpts[0][0])
            sign = np.dot(vs, fpts[0][1] - fpts[0][0])
            peri = len(fpts)*dist

        elif mode == 'rotation':
            # Rotation
            vs_mag = cfg.get('solver-moving-terms', 'rot-vel', '0')
            vs_mag = eval(re.sub(r'\b(pi)\b', 'np.pi', vs_mag))
            peri = 2.0*np.pi
            dist = peri/len(fpts)
            sign = 1.0

        if sign < 0: vs_mag = -vs_mag
        if ismv > 0.0: vs_mag = -vs_mag

        class SlideUnpackKernel(ComputeKernel):
            def run(self, queue, t=0):
                # Relative position (considering the uniform slide plane)
                move = vs_mag*t
                move -= peri*int(move/peri)

                # Move normalized by one edge
                n = move/dist

                # Movement on edge coordinate [-1,0] or [1,0]
                res = 2.0*(n - int(n))

                # How many edges to be passed
                n = int(n)

                # newfpts : new fpts coordinate [-1, 1]
                newfpts = [f - res for f in pts if f - res >= -1 and f - res <= 1.0]

                # newidx : index adjustment: it is tricky if n is negative
                newidx = (abs(n) + 1)*npts - len(newfpts)
                newfpts = newfpts + [f - res + 2.0 for f in pts if f - res + 2.0 < 1.0 and f - res + 2.0 > -1.0]
                newfpts = [f - res - 2.0 for f in pts if f - res - 2.0 < 1.0 and f - res - 2.0 > -1.0] + newfpts
                newfpts = np.array(sorted(newfpts))
                op = basis.nodal_basis_at(newfpts)

                # Access page-locked array for mpi communication
                m = mv.hdata.reshape(ioshape)

                for i in range(ioshape[0]):
                    for j in range(ioshape[1]):
                        tmp = m[i,j,:ioshape[-1]].reshape(npts,-1, order ='F')
                        tmp = np.dot(op, tmp)
                        tmp = tmp.reshape(-1, order = 'F')

                        if vs_mag < 0:
                            m[i,j,:] = np.concatenate([tmp[newidx:], tmp[0:newidx]])
                        else:
                            m[i,j,:] = np.concatenate([tmp[-newidx:], tmp[0:-newidx]])

                # Copy to device
                cuda.memcpy_htod_async(mv.data, mv.hdata,
                                       queue.cuda_stream_comp)

        return SlideUnpackKernel()