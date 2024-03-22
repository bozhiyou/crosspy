from parla import Parla, spawn, TaskSpace, parray
from parla.tasks import get_current_context
from parla.common.parray.coherence import Coherence
from parla.cython.device_manager import cpu, gpu
import numpy as np

def test_parray_concurrent_write():
    with Parla():
        @spawn(placement=cpu)
        def main():
            a = parray.asarray([1, 2, 3, 4])
            ts = TaskSpace("Concurrent write")

            @spawn(ts[1], placement=gpu(1), output=[(a,0)])
            def g1():
                assert a[0] == 1
                a.print_overview()
                a[0] = 5

            @spawn(ts[2], placement=gpu(0), output=[(a,0)])
            def g0():
                assert a[1] == 2
                a.print_overview()
                a[1] = 6

            @spawn(ts[3], dependencies=[ts[1], ts[2]], placement=cpu, output=[(a,0)])
            def c():
                print(a)
                a.print_overview()

def test_parray_nested_slicing():
    with Parla():
        @spawn(placement=cpu)
        def main():
            a = parray.asarray([[1, 2], [3, 4]])
            ts = TaskSpace("NestedSlicing")

            @spawn(ts[1], placement=gpu(1), output=[(a,0)])
            def g1():
                assert a[0,0] == 1
                a.print_overview()
                a[0,0] = 5

            @spawn(ts[2], dependencies=[ts[1]], placement=gpu(0), output=[(a,0)])
            def g0():
                b = a[:]
                assert b[0,0] == 5
                a.print_overview()



import crosspy as xp

class PArrayManager(xp.utils.wrapper.DynamicObjectManager):
    def wrap(self, array):
        return parray.asarray(array)
    
    def get_device(self, id):
        return get_current_context().devices[id]

def test_parray_creation():
    with Parla():
        # A = parray.asarray([[1, 2], [3, 4]])
        xA = xp.array([[1, 2], [3, 4]], data_manager=PArrayManager())

        a = xA[0]
        assert xA[0,1] == 2
        assert xA[1,0] == 3
        assert xA[1,1] == 4
        assert np.array_equal(xA, np.asarray([[1, 2], [3, 4]]))

def test_parray_task():
    with Parla():
        @spawn(placement=cpu)
        def main():
            np.random.seed(10)
            # Construct input data
            a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            b = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            a = parray.asarray(a, name="A")
            b = parray.asarray(b, name="B")

            # C++ test: parray parent id check
            c = b[0:10]
            a_parentid = a.get_parray_parentid_from_cpp()
            b_parentid = b.get_parray_parentid_from_cpp()
            assert c.get_parray_parentid_from_cpp() == b_parentid

            ts = TaskSpace("CopyBack")

            assert a._current_device_index == -1
            assert a._array._buffer[1] is None
            assert a._array._buffer[-1] is not None
            assert a._coherence._local_states[1] == Coherence.INVALID
            assert a._coherence._local_states[-1] == Coherence.MODIFIED

            @spawn(ts[1], placement=gpu(1), output=[(b,0)])
            def check_array_write():
                assert b[0,0] == 1
                assert b._current_device_index == 1
                b.print_overview()
                
                b[1,1] = 0
                assert b[1,1] == 0
                assert b._array._buffer[1] is not None
                assert b._array._buffer[-1] is None
                assert b._coherence._local_states[-1] == Coherence.INVALID
                assert b._coherence._local_states[1] == Coherence.MODIFIED

            @spawn(ts[2], dependencies=[ts[1]], placement=gpu(0), output=[(a[0:2],0), (a[2],0)])
            def check_array_slicing():
                assert a[1,0] == 1
                assert a._current_device_index == 0
                a[0:2].print_overview()
                assert a_parentid == a[0:2].get_parray_parentid_from_cpp()
                
                a[1,1] = 0
                assert a[1,1] == 0
                assert a._array._buffer[-1] is not None
                assert isinstance(a._array._buffer[0], list)
                assert a._coherence._local_states[-1] == Coherence.INVALID
                assert isinstance(a._coherence._local_states[0], dict)

            @spawn(ts[3], dependencies=[ts[2]], placement=cpu, output=[(a,0)])
            def check_array_write_back():
                assert a[1,1] == 0
                assert a._current_device_index == -1
                a.print_overview()
                
                assert a._array._buffer[-1] is not None
                assert a._array._buffer[0] is None
                assert a._coherence._local_states[-1] == Coherence.MODIFIED
                assert a._coherence._local_states[0] == Coherence.INVALID

            @spawn(ts[4], dependencies=[ts[3]], placement=gpu(1), inout=[(a,0)])
            def check_array_update():
                a.update(np.array([1,2,3,4]))  # here np array are converted to cupy array

                assert len(a) == 4
                assert a[-1] == 4
                assert a._coherence.owner == 1

            @spawn(ts[5], dependencies=[ts[4]], placement=gpu(1), inout=[(a,0)])
            def check_array_evict():
                a.print_overview()

                print(a)
                result = a.evict(1, False)

                assert result == False

                result = a.evict(1)

                assert result == True
                assert a._coherence.owner == -1
                assert a._array._buffer[1] is None

            @spawn(ts[6], dependencies=[ts[5]], placement=gpu(0), input=[(a,0)])
            def check_array_evict2():
                assert a._array._buffer[-1] is not None
                result = a.evict(-1, False)

                assert result == True

                assert a._coherence.owner == 0
                assert a._array._buffer[-1] is None

if __name__=="__main__":
    test_parray_concurrent_write()
    # test_parray_nested_slicing()
    # test_parray_creation()
    # test_parray_task()