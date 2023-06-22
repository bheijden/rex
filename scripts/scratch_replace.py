import flax.struct as struct
import jax.numpy as jp
import numpy as onp


@struct.dataclass
class Params:
	arr_jp: jp.ndarray
	arr_onp: onp.ndarray


if __name__ == "__main__":
	arr_jp = jp.array([1, 2, 3])
	arr_onp = onp.array([1, 2, 3])
	params = Params(arr_jp=arr_jp, arr_onp=arr_onp)

	new_arr_jp = jp.array([4, 5, 6])
	new_params = params.replace(arr_jp=new_arr_jp)

	arr_onp[:] = 999

	print(f"params: {params.arr_onp}")
	print(f"new_params: {new_params.arr_onp}")