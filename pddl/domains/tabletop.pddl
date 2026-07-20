;; ============================================================================
;; Domain: tabletop  (PDDL + PyTAMP TAMP pipeline, Phase 1)
;;
;; Symbolic pick / place / stack / unstack over a tabletop scene.
;;
;; GEOMETRIC BRIDGE PREDICATES — set by the PyTAMP geometric layer, NOT by the
;; symbolic planner: (reachable ?o) (graspable ?o) (stackable ?t ?b)
;; (placeable ?o ?s). The TAMP loop seeds these optimistically, then the
;; refiner falsifies them on geometric failure and the planner re-plans.
;;
;; Action signatures map 1:1 onto pytamp_moveit_bridge/TaskAction:
;;   action_type  <- action name (PICK/PLACE/STACK/UNSTACK)
;;   object_id    <- ?o / ?top
;;   preconditions/effects <- grounded literals below
;; ============================================================================
(define (domain tabletop)
  (:requirements :strips :typing :negative-preconditions :equality)

  (:types
    object surface gripper - locatable)

  (:predicates
    ;; --- symbolic state ---
    (on ?o - object ?s - surface)          ; o rests directly on surface s
    (stacked-on ?top - object ?bottom - object) ; top rests directly on bottom
    (clear ?o - object)                    ; nothing on top of o
    (holding ?g - gripper ?o - object)     ; gripper g is holding o
    (handempty ?g - gripper)               ; gripper g holds nothing
    ;; --- geometric bridge (set by PyTAMP) ---
    (reachable ?o - object)                ; IK solution exists to approach o
    (graspable ?o - object)                ; a collision-free grasp exists for o
    (stackable ?top - object ?bottom - object) ; top fits stably on bottom
    (placeable ?o - object ?s - surface))  ; a collision-free placement exists

  ;; Pick an object resting on a surface.
  (:action pick
    :parameters (?o - object ?s - surface ?g - gripper)
    :precondition (and (on ?o ?s) (clear ?o) (handempty ?g)
                       (reachable ?o) (graspable ?o))
    :effect (and (holding ?g ?o) (not (on ?o ?s))
                 (not (handempty ?g)) (not (clear ?o))))

  ;; Place a held object on a surface.
  (:action place
    :parameters (?o - object ?s - surface ?g - gripper)
    :precondition (and (holding ?g ?o) (placeable ?o ?s))
    :effect (and (on ?o ?s) (clear ?o) (handempty ?g)
                 (not (holding ?g ?o))))

  ;; Stack a held object on top of another (clear) object.
  (:action stack
    :parameters (?top - object ?bottom - object ?g - gripper)
    :precondition (and (holding ?g ?top) (clear ?bottom)
                       (stackable ?top ?bottom) (not (= ?top ?bottom)))
    :effect (and (stacked-on ?top ?bottom) (clear ?top) (handempty ?g)
                 (not (holding ?g ?top)) (not (clear ?bottom))))

  ;; Remove an object from the top of a stack.
  (:action unstack
    :parameters (?top - object ?bottom - object ?g - gripper)
    :precondition (and (stacked-on ?top ?bottom) (clear ?top) (handempty ?g)
                       (reachable ?top) (graspable ?top) (not (= ?top ?bottom)))
    :effect (and (holding ?g ?top) (clear ?bottom)
                 (not (stacked-on ?top ?bottom)) (not (handempty ?g))
                 (not (clear ?top))))
)
