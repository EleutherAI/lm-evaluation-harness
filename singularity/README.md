## Build

```
singularity build --fakeroot --nv --bind <path to tmp>:/tmp <path to store sif file> singularity/Singularity.def
```

## Exec

```
singularity exec --nv --containall --bind <lm eval path>:<lm eval path> [other singularity args] <path to store sif file> <command> [command args]
```
