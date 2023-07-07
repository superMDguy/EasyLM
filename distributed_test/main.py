# zhangqiaorjc@google.com

# python main.py --server_addr="10.164.0.19:1456" --num_hosts=2 --host_idx=0
# python main.py --server_addr="10.164.0.19:1456" --num_hosts=2 --host_idx=1
from absl import app
from absl import flags

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P, Mesh
from jax.experimental.pjit import pjit
 
flags.DEFINE_string('server_addr', '', help='server ip addr')
flags.DEFINE_integer('num_hosts', 1, help='num of hosts' )
flags.DEFINE_integer('host_idx', 0, help='index of current host' )
FLAGS = flags.FLAGS
 
 
def main(argv):
  print('running')
  jax.distributed.initialize(FLAGS.server_addr, FLAGS.num_hosts, FLAGS.host_idx)
  print('global devices=', jax.devices())
  print('local devices=', jax.local_devices())

  def f(x, w):
   return jnp.einsum('blm,md->bld', x, w)

  x = jnp.ones((2, 4, 24))
  w = jnp.ones((24, 4))
  print(f(x, w).shape)

  # Model parallelism via pjit
  n = jax.device_count()
  mesh_shape = (n,)
  device_mesh = np.array(jax.devices()).reshape(mesh_shape)
  with Mesh(device_mesh, ('mdl',)):
    result = pjit(f, in_axis_resources=(P(None, None, 'mdl'), P('mdl', None)), out_axis_resources=None)(x, w)
    print(result)
    
  # result is replicated on each chip
  print('print shapes of result on each chip locally')
  for i in range(len(result.device_buffers)):
    print(result.device_buffers[i].shape)
 
if __name__ == '__main__':
  app.run(main)