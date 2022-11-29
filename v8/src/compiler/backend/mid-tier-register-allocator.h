// Copyright 2020 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_COMPILER_BACKEND_MID_TIER_REGISTER_ALLOCATOR_H_
#define V8_COMPILER_BACKEND_MID_TIER_REGISTER_ALLOCATOR_H_

#include "src/base/compiler-specific.h"
#include "src/common/globals.h"
#include "src/compiler/backend/instruction.h"
#include "src/compiler/backend/register-allocation.h"
#include "src/flags/flags.h"
#include "src/utils/bit-vector.h"
#include "src/zone/zone-containers.h"
#include "src/zone/zone.h"

namespace v8 {
namespace internal {

class TickCounter;

namespace compiler {
class BlockState;
class VirtualRegisterData;
// The MidTierRegisterAllocator is a register allocator specifically designed to
// perform register allocation as fast as possible while minimizing spill moves.

class MidTierRegisterAllocationData final : public RegisterAllocationData {
 public:
  MidTierRegisterAllocationData(const RegisterConfiguration* config,
                                Zone* allocation_zone, Frame* frame,
                                InstructionSequence* code,
                                TickCounter* tick_counter,
                                const char* debug_name = nullptr);
  MidTierRegisterAllocationData(const MidTierRegisterAllocationData&) = delete;
  MidTierRegisterAllocationData& operator=(
      const MidTierRegisterAllocationData&) = delete;

  static MidTierRegisterAllocationData* cast(RegisterAllocationData* data) {
    DCHECK_EQ(data->type(), Type::kMidTier);
    return static_cast<MidTierRegisterAllocationData*>(data);
  }

  VirtualRegisterData& VirtualRegisterDataFor(int virtual_register);

  // Add a gap move between the given operands |from| and |to|.
  MoveOperands* AddGapMove(int instr_index, Instruction::GapPosition position,
                           const InstructionOperand& from,
                           const InstructionOperand& to);

  // Adds a gap move where both sides are PendingOperand operands.
  MoveOperands* AddPendingOperandGapMove(int instr_index,
                                         Instruction::GapPosition position);

  // Helpers to get a block from an |rpo_number| or |instr_index|.
  const InstructionBlock* GetBlock(const RpoNumber rpo_number);
  const InstructionBlock* GetBlock(int instr_index);

  // Returns a bitvector representing all the blocks that are dominated by the
  // output of the instruction in |block|.
  const BitVector* GetBlocksDominatedBy(const InstructionBlock* block);

  // List of all instruction indexs that require a reference map.
  ZoneVector<int>& reference_map_instructions() {
    return reference_map_instructions_;
  }

  // Returns a bitvector representing the virtual registers that were spilled.
  BitVector& spilled_virtual_registers() { return spilled_virtual_registers_; }

  // This zone is for data structures only needed during register allocation
  // phases.
  Zone* allocation_zone() const { return allocation_zone_; }

  // This zone is for InstructionOperands and moves that live beyond register
  // allocation.
  Zone* code_zone() const { return code()->zone(); }

  BlockState& block_state(RpoNumber rpo_number);

  InstructionSequence* code() const { return code_; }
  Frame* frame() const { return frame_; }
  const char* debug_name() const { return debug_name_; }
  const RegisterConfiguration* config() const { return config_; }
  TickCounter* tick_counter() { return tick_counter_; }

 private:
  Zone* const allocation_zone_;
  Frame* const frame_;
  InstructionSequence* const code_;
  const char* const debug_name_;
  const RegisterConfiguration* const config_;

  ZoneVector<VirtualRegisterData> virtual_register_data_;
  ZoneVector<BlockState> block_states_;
  ZoneVector<int> reference_map_instructions_;
  BitVector spilled_virtual_registers_;

  TickCounter* const tick_counter_;
};

// Phase 1: Process instruction outputs to determine how each virtual register
// is defined.
void DefineOutputs(MidTierRegisterAllocationData* data);

// Phase 2: Allocate registers to instructions.
void AllocateRegisters(MidTierRegisterAllocationData* data);

// Phase 3: assign spilled operands to specific spill slots.
void AllocateSpillSlots(MidTierRegisterAllocationData* data);

// Phase 4: Populate reference maps for spilled references.
void PopulateReferenceMaps(MidTierRegisterAllocationData* data);

class RegisterState;
class DeferredBlocksRegion;

// BlockState stores details associated with a particular basic block.
class BlockState final {
 public:
  BlockState(int block_count, Zone* zone)
      : general_registers_in_state_(nullptr),
        double_registers_in_state_(nullptr),
        deferred_blocks_region_(nullptr),
        dominated_blocks_(block_count, zone),
        successors_phi_index_(-1),
        is_deferred_block_boundary_(false) {}

  // Returns the RegisterState that applies to the input of this block. Can be
  // |nullptr| if the no registers of |kind| have been allocated up to this
  // point.
  RegisterState* register_in_state(RegisterKind kind);
  void set_register_in_state(RegisterState* register_state, RegisterKind kind);

  // Returns a bitvector representing all the basic blocks that are dominated
  // by this basic block.
  BitVector* dominated_blocks() { return &dominated_blocks_; }

  // Set / get this block's index for successor's phi operations. Will return
  // -1 if this block has no successor's with phi operations.
  int successors_phi_index() const { return successors_phi_index_; }
  void set_successors_phi_index(int index) {
    DCHECK_EQ(successors_phi_index_, -1);
    successors_phi_index_ = index;
  }

  // If this block is deferred, this represents region of deferred blocks
  // that are directly reachable from this block.
  DeferredBlocksRegion* deferred_blocks_region() const {
    return deferred_blocks_region_;
  }
  void set_deferred_blocks_region(DeferredBlocksRegion* region) {
    DCHECK_NULL(deferred_blocks_region_);
    deferred_blocks_region_ = region;
  }

  // Returns true if this block represents either a transition from
  // non-deferred to deferred or vice versa.
  bool is_deferred_block_boundary() const {
    return is_deferred_block_boundary_;
  }
  void MarkAsDeferredBlockBoundary() { is_deferred_block_boundary_ = true; }

  MOVE_ONLY_NO_DEFAULT_CONSTRUCTOR(BlockState);

 private:
  RegisterState* general_registers_in_state_;
  RegisterState* double_registers_in_state_;
  RegisterState* simd128_registers_in_state_;

  DeferredBlocksRegion* deferred_blocks_region_;

  BitVector dominated_blocks_;
  int successors_phi_index_;
  bool is_deferred_block_boundary_;
};

// A Range from [start, end] of instructions, inclusive of start and end.
class Range {
 public:
  Range() : start_(kMaxInt), end_(0) {}
  Range(int start, int end) : start_(start), end_(end) {}

  void AddInstr(int index) {
    start_ = std::min(start_, index);
    end_ = std::max(end_, index);
  }

  void AddRange(const Range& other) {
    start_ = std::min(start_, other.start_);
    end_ = std::max(end_, other.end_);
  }

  // Returns true if index is greater than start and less than or equal to end.
  bool Contains(int index) { return index >= start_ && index <= end_; }

  int start() const { return start_; }
  int end() const { return end_; }

 private:
  int start_;
  int end_;
};

// Represents a connected region of deferred basic blocks.
class DeferredBlocksRegion final {
 public:
  explicit DeferredBlocksRegion(Zone* zone, int number_of_blocks)
      : spilled_vregs_(zone),
        blocks_covered_(number_of_blocks, zone),
        is_frozen_(false) {}

  void AddBlock(RpoNumber block, MidTierRegisterAllocationData* data) {
    DCHECK(data->GetBlock(block)->IsDeferred());
    blocks_covered_.Add(block.ToInt());
    data->block_state(block).set_deferred_blocks_region(this);
  }

  // Trys to adds |vreg| to the list of variables to potentially defer their
  // output to a spill slot until we enter this deferred block region. Returns
  // true if successful.
  bool TryDeferSpillOutputUntilEntry(int vreg) {
    if (spilled_vregs_.count(vreg) != 0) return true;
    if (is_frozen_) return false;
    spilled_vregs_.insert(vreg);
    return true;
  }

  void FreezeDeferredSpills() { is_frozen_ = true; }

  ZoneSet<int>::const_iterator begin() const { return spilled_vregs_.begin(); }
  ZoneSet<int>::const_iterator end() const { return spilled_vregs_.end(); }

  const BitVector* blocks_covered() const { return &blocks_covered_; }

 private:
  ZoneSet<int> spilled_vregs_;
  BitVector blocks_covered_;
  bool is_frozen_;
};

// VirtualRegisterData stores data specific to a particular virtual register,
// and tracks spilled operands for that virtual register.
class VirtualRegisterData final {
 public:
  VirtualRegisterData() = default;

  // Define VirtualRegisterData with the type of output that produces this
  // virtual register.
  void DefineAsUnallocatedOperand(int virtual_register,
                                  MachineRepresentation rep, int instr_index,
                                  bool is_deferred_block,
                                  bool is_exceptional_call_output);
  void DefineAsFixedSpillOperand(AllocatedOperand* operand,
                                 int virtual_register,
                                 MachineRepresentation rep, int instr_index,
                                 bool is_deferred_block,
                                 bool is_exceptional_call_output);
  void DefineAsConstantOperand(ConstantOperand* operand,
                               MachineRepresentation rep, int instr_index,
                               bool is_deferred_block);
  void DefineAsPhi(int virtual_register, MachineRepresentation rep,
                   int instr_index, bool is_deferred_block);

  // Spill an operand that is assigned to this virtual register.
  void SpillOperand(InstructionOperand* operand, int instr_index,
                    bool has_constant_policy,
                    MidTierRegisterAllocationData* data);

  // Emit gap moves to / from the spill slot.
  void EmitGapMoveToInputFromSpillSlot(InstructionOperand to_operand,
                                       int instr_index,
                                       MidTierRegisterAllocationData* data);
  void EmitGapMoveFromOutputToSpillSlot(InstructionOperand from_operand,
                                        const InstructionBlock* current_block,
                                        int instr_index,
                                        MidTierRegisterAllocationData* data);
  void EmitGapMoveToSpillSlot(InstructionOperand from_operand, int instr_index,
                              MidTierRegisterAllocationData* data);

  // Adds pending spills for deferred-blocks.
  void AddDeferredSpillUse(int instr_index,
                           MidTierRegisterAllocationData* data);
  void AddDeferredSpillOutput(AllocatedOperand allocated_op, int instr_index,
                              MidTierRegisterAllocationData* data);

  // Accessors for spill operand, which may still be pending allocation.
  bool HasSpillOperand() const { return spill_operand_ != nullptr; }
  InstructionOperand* spill_operand() const {
    DCHECK(HasSpillOperand());
    return spill_operand_;
  }

  bool HasPendingSpillOperand() const {
    return HasSpillOperand() && spill_operand_->IsPending();
  }
  bool HasAllocatedSpillOperand() const {
    return HasSpillOperand() && spill_operand_->IsAllocated();
  }
  bool HasConstantSpillOperand() const {
    return HasSpillOperand() && spill_operand_->IsConstant();
  }

  // Returns true if the virtual register should be spilled when it is output.
  bool NeedsSpillAtOutput() const { return needs_spill_at_output_; }

  void MarkAsNeedsSpillAtOutput() {
    if (HasConstantSpillOperand()) return;
    needs_spill_at_output_ = true;
    if (HasSpillRange()) spill_range()->ClearDeferredBlockSpills();
  }

  // Returns true if the virtual register should be spilled at entry to deferred
  // blocks in which it is spilled (to avoid spilling on output on
  // non-deferred blocks).
  bool NeedsSpillAtDeferredBlocks() const;
  void EmitDeferredSpillOutputs(MidTierRegisterAllocationData* data);

  bool IsSpilledAt(int instr_index, MidTierRegisterAllocationData* data) {
    DCHECK_GE(instr_index, output_instr_index());
    if (NeedsSpillAtOutput() || HasConstantSpillOperand()) return true;
    if (HasSpillOperand() && data->GetBlock(instr_index)->IsDeferred()) {
      return true;
    }
    return false;
  }

  // Allocates pending spill operands to the |allocated| spill slot.
  void AllocatePendingSpillOperand(const AllocatedOperand& allocated);

  int vreg() const { return vreg_; }
  MachineRepresentation rep() const { return rep_; }
  int output_instr_index() const { return output_instr_index_; }
  bool is_constant() const { return is_constant_; }
  bool is_phi() const { return is_phi_; }
  bool is_defined_in_deferred_block() const {
    return is_defined_in_deferred_block_;
  }
  bool is_exceptional_call_output() const {
    return is_exceptional_call_output_;
  }

  struct DeferredSpillSlotOutput {
   public:
    explicit DeferredSpillSlotOutput(int instr, AllocatedOperand op,
                                     const BitVector* blocks)
        : instr_index(instr), operand(op), live_blocks(blocks) {}

    int instr_index;
    AllocatedOperand operand;
    const BitVector* live_blocks;
  };

  // Represents the range of instructions for which this virtual register needs
  // to be spilled on the stack.
  class SpillRange : public ZoneObject {
   public:
    // Defines a spill range for an output operand.
    SpillRange(int definition_instr_index,
               const InstructionBlock* definition_block,
               MidTierRegisterAllocationData* data)
        : live_range_(definition_instr_index, definition_instr_index),
          live_blocks_(data->GetBlocksDominatedBy(definition_block)),
          deferred_spill_outputs_(nullptr) {}

    // Defines a spill range for a Phi variable.
    SpillRange(const InstructionBlock* phi_block,
               MidTierRegisterAllocationData* data)
        : live_range_(phi_block->first_instruction_index(),
                      phi_block->first_instruction_index()),
          live_blocks_(data->GetBlocksDominatedBy(phi_block)),
          deferred_spill_outputs_(nullptr) {
      // For phis, add the gap move instructions in the predecssor blocks to
      // the live range.
      for (RpoNumber pred_rpo : phi_block->predecessors()) {
        const InstructionBlock* block = data->GetBlock(pred_rpo);
        live_range_.AddInstr(block->last_instruction_index());
      }
    }

    SpillRange(const SpillRange&) = delete;
    SpillRange& operator=(const SpillRange&) = delete;

    bool IsLiveAt(int instr_index, InstructionBlock* block) {
      if (!live_range_.Contains(instr_index)) return false;

      int block_rpo = block->rpo_number().ToInt();
      if (!live_blocks_->Contains(block_rpo)) return false;

      if (!HasDeferredBlockSpills()) {
        return true;
      } else {
        // If this spill range is only output for deferred block, then the spill
        // slot will only be live for the deferred blocks, not all blocks that
        // the virtual register is live.
        for (auto deferred_spill_output : *deferred_spill_outputs()) {
          if (deferred_spill_output.live_blocks->Contains(block_rpo)) {
            return true;
          }
        }
        return false;
      }
    }

    void ExtendRangeTo(int instr_index) { live_range_.AddInstr(instr_index); }

    void AddDeferredSpillOutput(AllocatedOperand allocated_op, int instr_index,
                                MidTierRegisterAllocationData* data) {
      if (deferred_spill_outputs_ == nullptr) {
        Zone* zone = data->allocation_zone();
        deferred_spill_outputs_ =
            zone->New<ZoneVector<DeferredSpillSlotOutput>>(zone);
      }
      const InstructionBlock* block = data->GetBlock(instr_index);
      DCHECK_EQ(block->first_instruction_index(), instr_index);
      BlockState& block_state = data->block_state(block->rpo_number());
      const BitVector* deferred_blocks =
          block_state.deferred_blocks_region()->blocks_covered();
      deferred_spill_outputs_->emplace_back(instr_index, allocated_op,
                                            deferred_blocks);
    }

    void ClearDeferredBlockSpills() { deferred_spill_outputs_ = nullptr; }
    bool HasDeferredBlockSpills() const {
      return deferred_spill_outputs_ != nullptr;
    }
    const ZoneVector<DeferredSpillSlotOutput>* deferred_spill_outputs() const {
      DCHECK(HasDeferredBlockSpills());
      return deferred_spill_outputs_;
    }

    Range& live_range() { return live_range_; }

   private:
    Range live_range_;
    const BitVector* live_blocks_;
    ZoneVector<DeferredSpillSlotOutput>* deferred_spill_outputs_;
  };

  bool HasSpillRange() const { return spill_range_ != nullptr; }
  SpillRange* spill_range() const {
    DCHECK(HasSpillRange());
    return spill_range_;
  }

 private:
  void Initialize(int virtual_register, MachineRepresentation rep,
                  InstructionOperand* spill_operand, int instr_index,
                  bool is_phi, bool is_constant,
                  bool is_defined_in_deferred_block,
                  bool is_exceptional_call_output);

  void AddSpillUse(int instr_index, MidTierRegisterAllocationData* data);
  void AddPendingSpillOperand(PendingOperand* pending_operand);
  void EnsureSpillRange(MidTierRegisterAllocationData* data);
  bool TrySpillOnEntryToDeferred(MidTierRegisterAllocationData* data,
                                 const InstructionBlock* block);

  InstructionOperand* spill_operand_;
  SpillRange* spill_range_;
  int output_instr_index_;

  int vreg_;
  MachineRepresentation rep_;
  bool is_phi_ : 1;
  bool is_constant_ : 1;
  bool is_defined_in_deferred_block_ : 1;
  bool needs_spill_at_output_ : 1;
  bool is_exceptional_call_output_ : 1;
};

}  // namespace compiler
}  // namespace internal
}  // namespace v8

#endif  // V8_COMPILER_BACKEND_MID_TIER_REGISTER_ALLOCATOR_H_
