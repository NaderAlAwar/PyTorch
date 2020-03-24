#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/future.h>

#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

namespace callback {
// It's the callback for RemoteCall.
void TORCH_API confirmPendingUser(
    const rpc::Message& message,
    const c10::optional<utils::FutureError>& futErr);

// It's the callback for finishing creating owner rref, it returned deletedRRef,
// so that the deletedRRef can be handled under GIL in python_functions.cpp if
// deletedRRef contains python object.
c10::intrusive_ptr<RRef> TORCH_API finishCreatingOwnerRRef(
    const Message& message,
    const c10::optional<utils::FutureError>& futErr);
} // namespace callback

// Manages RRef lifetime and keeps track of RRef forks.
class TORCH_API RRefContext {
 public:
  static RRefContext& getInstance();
  // NB: This method must be called before destructing RRefContext singleton.
  // Similar to delForkOfOwner, this method returns a vector of OwnerRRefs that
  // hold py::object. The call-site is also responsible for resetting those
  // shared_ptr objects with a GIL. See comments at delForkOfOwner() for more
  // details.
  static std::vector<c10::intrusive_ptr<RRef>> destroyInstance(
      bool ignoreRRefLeak = true);

  static void handleException(const c10::optional<utils::FutureError>& futErr);

  RRefContext(const RRefContext&) = delete;
  RRefContext(RRefContext&& other) = delete;
  void operator=(const RRefContext&) = delete;
  RRefContext& operator=(RRefContext&& other) = delete;

  ~RRefContext();

  // get the worker id of the current worker
  inline worker_id_t getWorkerId() const {
    return agent_->getWorkerInfo().id_;
  }

  // get the worker name of the current worker
  inline const std::string& getWorkerName() const {
    return agent_->getWorkerInfo().name_;
  }

  //  generate a globally unique ID
  inline GloballyUniqueId genGloballyUniqueId() {
    return GloballyUniqueId(getWorkerId(), nextLocalId_++);
  }

  inline const std::shared_ptr<RpcAgent>& agent() const {
    return agent_;
  }

  // create a ``UserRRef`` owned by the worker ``ownerId``
  c10::intrusive_ptr<UserRRef> createUserRRef(
      worker_id_t ownerId,
      const TypePtr& type);

  // Convert an RRefForkData into an RRef. This RRef could be user or owner.
  // This RRef could have already existed before, or could be created in this
  // method, we pass type here to validate or help the rref creation.
  c10::intrusive_ptr<RRef> getOrCreateRRef(
      const RRefForkData& rfd,
      const TypePtr& type);

  // Get the ``OwnerRRef`` of id ``rrefId``. If it does not exist, create a new
  // one.
  c10::intrusive_ptr<OwnerRRef> getOrCreateOwnerRRef(
      const RRefId& rrefId,
      const TypePtr& type);

  // Create an empty owner rref of type.
  c10::intrusive_ptr<OwnerRRef> createOwnerRRef(const TypePtr& type);

  c10::intrusive_ptr<OwnerRRef> getOwnerRRef(const RRefId& rrefId);

  // Adding the RRefId of an OwnerRRef into the forks_ map. This is useful when
  // making a remote call to self, which as for now, still goes through serde
  // and invokes request callback. In this case, the OwnerRRef has already been
  // created on the send side, and we need to pass it to the receive side,
  // instead of creating a new OwnerRRef. This is done by adding the OwnerRRef
  // into owners_. However, that alone is not enough, as it could be deleted
  // when all UserRRef die, which would then remove the OwnerRRef from owners_
  // and this could happen before the self remote call finishes. To prevent
  // that, this API adds the RRefId as a ForkId, which will then delete the
  // ForkId when the self remote is done.
  void addSelfAsFork(c10::intrusive_ptr<OwnerRRef>& rref);

  // Register a fork of the ``OwnerRRef``, and inserts a intrusive_ptr of the
  // ``OwnerRRef`` in a map to keep it alive.
  void addForkOfOwner(const RRefId& rrefId, const ForkId& forkId);
  // Performs the same function as addForkOfOwner but ignores duplicate
  // requests. This idempotent function is used with RREF_FORK_REQUEST calls,
  // whereas all other message types use the non-idempotent variant.
  void addForkOfOwnerIfNotPresent(const RRefId& rrefId, const ForkId& forkId);
  // Delete a fork of the ``OwnerRRef``. NB: this could trigger deletion on the
  // IValue or py::object. For the later, this method will acquire GIL.
  // NB: If this fork deletion triggered deleting OwnerRRef, this method will
  // return a shared_ptr to the OwnerRRef, which is likely to be the last
  // shared_ptr instance for it. Therefore, deleting this shared_ptr<OwnerRRef>
  // will also trigger deleting the object it points to. If OwnerRRef holds a
  // py::object, deleting it require GIL. The call site should guarded it with
  // a GIL and reset the shared_ptr. The GIL-guarded deletion is intentionally
  // left out of this function to avoid creating dependency on pybind.
  c10::intrusive_ptr<RRef> delForkOfOwner(
      const RRefId& rrefId,
      const ForkId& forkId);

  // Invoked when pickling an RRef to setup child/fork properly
  RRefForkData prepareChildFork(const c10::intrusive_ptr<RRef>& rref);
  // Invoked when unpickling an RRef to send RREF_FORK_REQUEST to owner and
  // send RREF_CHILD_ACCEPT to the parent.
  // NB: forkId is necessary here as the rref could be an OwnerRRef
  void notifyOwnerAndParentOfFork(
      const ForkId& forkId,
      worker_id_t parent,
      const c10::intrusive_ptr<RRef>& rref);

  // When a UserRRef is forked to another worker (user or owner), it is added
  // into pendingChildren_ to be held alive until it receives RREF_CHILD_ACCEPT
  // from the child.
  // NB: This is necessary for both user and owner child. As we do not have FIFO
  // communication between workers, we need this strategy to make sure that all
  // previously submitted rpc/remote calls are acked before sending out the
  // RREF_USER_DELETE message. Otherwise, the OwnerRRef could be deleted too
  // soon.
  void addPendingChild(
      const ForkId& forkId,
      const c10::intrusive_ptr<RRef>& rref);
  void delPendingChild(const ForkId& forkId);

  // When a UserRRef is created, it is added into pendingUsers_ to be held alive
  // until it receives RREF_USER_ACCEPT from the owner.
  void addPendingUser(
      const ForkId& forkId,
      const c10::intrusive_ptr<RRef>& rref);
  void delPendingUser(const ForkId& forkId);
  void addConfirmedUser(
      const ForkId& forkId,
      const c10::intrusive_ptr<RRef>& rref);

  // Start recroding new pending UserRRefs. All pending UserRRefs introduced
  // after this point will be put into the thread_local userTable_, which will
  // then be consumed and cleared in waitForThreadLocalPendingRRefs().
  void recordThreadLocalPendingRRefs();
  // End recording new pending UserRRefs, and clear the thread_local userTable_.
  // Returns a Future which will be marked as completed when all pending
  // UserRRefs in the current userTable_ are confirmed by their owners. The bool
  // value in the Future is unused.
  // This method is useful to make sure RRefs in user function arguments are
  // confirmed before launching user code.
  // NB: Callers of this method does not need to keep the returned Future alive,
  // because this Future is already captured in callbacks of the
  // PendingUserState. If there is no pending UserRRefs, this method returns a
  // completed future.
  std::shared_ptr<torch::utils::Future<bool>> waitForThreadLocalPendingRRefs();
  // Only call this function when there are errors during a recording session,
  // and it is likely that waitForThreadLocalPendingRRefs() cannot be invoked
  // properly.
  // TODO: make this a context guard
  void clearRecordedPendingRRefsOnError();

  void delUser(
      const worker_id_t owner,
      const RRefId& rrefId,
      const ForkId& forkId);
  void delAllUsers(std::chrono::milliseconds timeoutMillis);

  std::unordered_map<std::string, std::string> getDebugInfo();

 private:
  struct PendingUserState {
    PendingUserState(c10::intrusive_ptr<RRef> rref) : rref_(std::move(rref)) {}

    inline void confirm() {
      c10::static_intrusive_pointer_cast<UserRRef>(rref_)->confirm();
      future_.markCompleted(true);
    }

    c10::intrusive_ptr<RRef> rref_;
    // Use Future.wait() and Future.markCompleted() to block and unblock user
    // functions. The bool value wrapped by the future_ is not used.
    torch::utils::Future<bool> future_;
  };

  RRefContext(std::shared_ptr<RpcAgent>);

  c10::intrusive_ptr<UserRRef> createUserRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId,
      const TypePtr& type);

  void finishForkRequest(const ForkId& forkId, worker_id_t parent);

  // If there is any leak on any RRef, this method will throw an error.
  void checkRRefLeaks(bool ignoreRRefLeak);

  static std::atomic<local_id_t> nextLocalId_;

  const std::shared_ptr<RpcAgent> agent_;
  mutable std::mutex mutex_;
  // Keep OwnerRRefs alive until there is no living UserRRefs.
  std::unordered_map<RRefId, c10::intrusive_ptr<RRef>, RRefId::Hash> owners_;
  // A conditional variable to block getOwnerRRef() calls until the
  // corresponding OwnerRRef has been created and inserted into the owners_ map.
  // The method getOwnerRRef() is triggered by rref.to_here() messages. The
  // reason for having this CV is because rref.to_here() message and rpc.remote
  // message may arrive in any order, and to_here() can only be served when the
  // RRef value is ready. In the previous version, we used to block the
  // to_here() call by waiting on the CV member variable in OwnerRRef. However,
  // that means the to_here() call has to first create the OwnerRRef, which
  // would require knowing the IValue type when if we want to make RRef an
  // IValue. Instead of sending serialized TypePtr in every to_here() message,
  // we decided to only create OwnerRRef when processing remote calls.
  // TODO: As OwnerRRef::getValue() is always called after
  // OwnerRRef::setValue(), we should be able to remove the CV from OwnerRRef.
  std::condition_variable ownerCV_;
  // Tracks known living UserRRefs of an OwnerRRef
  std::unordered_map<
      RRefId,
      std::unordered_set<ForkId, ForkId::Hash>,
      RRefId::Hash>
      forks_;

  // This cond var is used by deleteAllUsers(), a event notificaton is sent if
  // number of pending UserRRef or UserRRef children is reduced, or
  // number of owned OwnerRRef is reduced.
  std::condition_variable deleteAllUsersCV_;
  // The follow 3 maps keep UserRRefs alive by holding a intrusive_ptr to the
  // RRef instances. A UserRRef must be added into this map if any of the
  // following two conditions is true:
  //
  // (1) A UserRRef has not been accepted by owner yet.
  //
  //     It can be used or shared, but cannot be deleted, and hence kept alive
  //     in this map. A message of type RREF_USER_ACCEPT will move the
  //     corresponding RRef from pendingUsers_ map to confirmedUsers_ map.
  std::unordered_map<ForkId, std::shared_ptr<PendingUserState>, ForkId::Hash>
      pendingUsers_;
  //     UserRRefs are added into this map when it is confirmed by the owner.
  //     When destroying RRefContext this map helps to find local UserRRefs
  //     and send delete messages if they are still not deleted by Python
  //     garbage collection.
  std::unordered_map<ForkId, c10::weak_intrusive_ptr<RRef>, ForkId::Hash>
      confirmedUsers_;

  // (2) A UserRRef has forked a child UserRRef which has not been accepted by
  //     the owner yet.
  //
  //     In this case, this UserRRef cannot send out RREF_USER_DELETE message,
  //     as it could potentially trigger the OwnerRRef been deleted before the
  //     owner learns about the forked child.
  std::unordered_map<ForkId, c10::intrusive_ptr<RRef>, ForkId::Hash>
      pendingChildren_;

  std::mutex destroyedMutex_;
  bool destroyed_;

  // Thread local states to keep UserRRefs deserialized from user function
  // arguments.
  static thread_local std::vector<std::shared_ptr<PendingUserState>> userTable_;
  // A flag indicating whether subsequently created UserRRefs should be added to
  // the thread_local userTable_. The flag is set to true before serializing
  // RPC arguments and then set to false before running the corresponding
  // user code. See addPendingUser and delPendingUser for more details.
  // NB: The reason for having this flag is because addPendingUser are called in
  // two cases, and we only want to track the 2nd case.
  // (1) RRef as the return value: when calling rpc.remote, the UserRRef on the
  //     caller side is added to the context using addPendingUser.
  // (2) RRef as an argument: When running an RPC using RRefs as arguments, the
  //     RRef is forwarded to the callee as new UserRRefs (if the callee is not
  //     the owner). In this case, we block running the user function until all
  //     UserRRefs are confirmed by the owner.
  // This contract gurantees that no UserRRefs can be used remotely without
  // confirmation. Note that, however, the UserRRef created by rpc.remote can
  // still be passed to local functions as arguments and used there. This is by
  // design, because this feature is especially useful when, say a master node
  // creates multiple UserRRefs in a loop and then shares them with other nodes.
  // Blocking every iteration in the loop until RRefs are confirmed will slow
  // this down. This nuance on UserRRef can be interpreted as we only make
  // exceptions for UserRRef creators. And using the UserRRef on its creator
  // without confirmation is OK, because the creator would either call to_here
  // or forward the UserRRef, and both would then require confirmations from the
  // owner.
  static thread_local bool recording;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
