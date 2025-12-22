from animo_io.embedding.hash_embedder import HashTextEmbedder
from animo_io.skeletons.toy_dog import ToySkeletonProvider
from animo_io.generators.dummy_generator import DummyMotionGenerator

def main():
    embedder = HashTextEmbedder(dim=256)
    skel_provider = ToySkeletonProvider()
    gen = DummyMotionGenerator(embedder=embedder, skeletons=skel_provider)

    out = gen.generate("犬が歩く / a dog is walking", species="dog", T=120)

    print("=== Skeleton ===")
    print("J =", out.skeleton.J)
    print("joint_names =", out.skeleton.joint_names)
    print("parents     =", out.skeleton.parents)
    print("rest_offsets shape =", out.skeleton.rest_offsets.shape)

    print("\n=== Motion ===")
    print("T =", out.motion.T)
    print("root_translation shape =", out.motion.root_translation.shape)  # (T,3)
    print("joint_quat shape       =", out.motion.joint_quat.shape)        # (T,J,4)
    print("foot_contacts shape    =", out.motion.foot_contacts.shape)     # (T,2)

if __name__ == "__main__":
    main()
