(define (problem elevators-time-p8_7_1) (:domain elevators-time)
  (:init
  
    (can-start-rewards)

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;  Elevators
  
  ;; Initial Condition
  	(passengers slow0-0 n0)
  	(passengers slow1-0 n0)
  	(passengers fast1 n0)
  	(passengers fast0 n0)
    
  	(lift-at slow0-0 n0)
  	(lift-at slow1-0 n7)
  	(lift-at fast1 n2)
  	(lift-at fast0 n2)
    
  	(can-hold slow0-0 n1)
  	(can-hold slow0-0 n2)
  	(can-hold slow1-0 n1)
  	(can-hold slow1-0 n2)
  	(can-hold fast1 n1)
  	(can-hold fast1 n2)
  	(can-hold fast1 n3)
  	(can-hold fast0 n1)
  	(can-hold fast0 n2)
  	(can-hold fast0 n3)

    
  	(reachable-floor slow0-0 n0)
  	(reachable-floor slow0-0 n1)
  	(reachable-floor slow0-0 n2)
  	(reachable-floor slow0-0 n3)
  	(reachable-floor slow0-0 n4)

  	(reachable-floor slow1-0 n4)
  	(reachable-floor slow1-0 n5)
  	(reachable-floor slow1-0 n6)
  	(reachable-floor slow1-0 n7)
  	(reachable-floor slow1-0 n8)

  	(reachable-floor fast0 n0)
  	(reachable-floor fast0 n4)
  	(reachable-floor fast0 n2)
  	(reachable-floor fast0 n6)
  	(reachable-floor fast0 n8)
    
  	(reachable-floor fast1 n0)
  	(reachable-floor fast1 n2)
  	(reachable-floor fast1 n4)
  	(reachable-floor fast1 n6)
  	(reachable-floor fast1 n8)

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Passenger
  
  	(passenger-at p6 n0)
  	(passenger-at p2 n1)
  	(passenger-at p0 n2)
  	(passenger-at p3 n3)
  	(passenger-at p1 n6)
  	(passenger-at p5 n6)
  	(passenger-at p4 n6)


  	(goal-passenger-at p0 n8)
  	(goal-passenger-at p1 n1)
  	(goal-passenger-at p2 n7)
  	(goal-passenger-at p3 n0)
  	(goal-passenger-at p4 n0)
  	(goal-passenger-at p5 n5)
  	(goal-passenger-at p6 n5)
    
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;  Floor Construction
  
	(above n0 n2)
	(above n0 n1)
	(above n0 n3)
	(above n0 n4)
	(above n0 n5)
	(above n0 n6)
	(above n0 n7)
	(above n0 n8)
    
  	(above n1 n2)
  	(above n1 n3)
  	(above n1 n4)
  	(above n1 n5)
  	(above n1 n6)
  	(above n1 n7)
  	(above n1 n8)
    
  	(above n2 n8)
  	(above n2 n7)
  	(above n2 n6)
  	(above n2 n5)
  	(above n2 n3)
  	(above n2 n4)

  	(above n3 n6)
  	(above n3 n7)
  	(above n3 n8)
  	(above n3 n5)
  	(above n3 n4)

  	(above n4 n5)
  	(above n4 n6)
  	(above n4 n8)
  	(above n4 n7)

  	(above n5 n6)
  	(above n5 n7)
  	(above n5 n8)

  	(above n6 n7)
  	(above n6 n8)
    
  	(above n7 n8)


  ;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;  Numerical Properties
  
  	(next n7 n8)
  	(next n6 n7)
  	(next n5 n6)
  	(next n4 n5)
  	(next n3 n4)
  	(next n2 n3)
  	(next n1 n2)
  	(next n0 n1)
    
    
  	(= (travel-slow n5 n8) 28.0)
  	(= (travel-fast n2 n4) 11.0)
  	(= (travel-slow n6 n7) 12.0)
  	(= (travel-fast n2 n8) 15.0)
  	(= (travel-slow n4 n6) 20.0)
  	(= (travel-slow n1 n4) 28.0)
  	(= (travel-fast n4 n6) 11.0)
  	(= (travel-fast n2 n6) 13.0)
  	(= (travel-fast n6 n8) 11.0)
  	(= (travel-slow n0 n1) 12.0)
  	(= (travel-fast n0 n6) 15.0)
  	(= (travel-slow n0 n2) 20.0)
  	(= (travel-fast n0 n2) 11.0)
  	(= (travel-slow n5 n6) 12.0)
  	(= (travel-fast n4 n8) 13.0)
  	(= (travel-fast n0 n4) 13.0)
  	(= (travel-slow n2 n4) 20.0)
  	(= (travel-slow n2 n3) 12.0)
  	(= (travel-slow n5 n7) 20.0)
  	(= (travel-slow n3 n4) 12.0)
  	(= (travel-slow n0 n3) 28.0)
  	(= (travel-slow n7 n8) 12.0)
  	(= (travel-slow n4 n8) 36.0)
  	(= (travel-fast n0 n8) 17.0)
  	(= (travel-slow n4 n5) 12.0)
  	(= (travel-slow n1 n2) 12.0)
  	(= (travel-slow n0 n4) 36.0)
  	(= (travel-slow n1 n3) 20.0)
  	(= (travel-slow n4 n7) 28.0)
  	(= (travel-slow n6 n8) 20.0)

  	(= (flattime-goal-passenger-at p6 n5) 2569.4087)
  	(= (flattime-goal-passenger-at p2 n7) 3762.213)
  	(= (flattime-goal-passenger-at p5 n5) 1228.5906)
  	(= (flattime-goal-passenger-at p3 n0) 2333.3342)
  	(= (flattime-goal-passenger-at p1 n1) 2542.175)
  	(= (flattime-goal-passenger-at p0 n8) 1539.101)
  	(= (flattime-goal-passenger-at p4 n0) 1538.902)
    
  	(= (slopetime-goal-passenger-at p4 n0) 38.90203)
  	(= (slopetime-goal-passenger-at p3 n0) 33.334164)
  	(= (slopetime-goal-passenger-at p6 n5) 69.40872)
  	(= (slopetime-goal-passenger-at p5 n5) 28.590628)
  	(= (slopetime-goal-passenger-at p2 n7) 62.212963)
  	(= (slopetime-goal-passenger-at p1 n1) 42.175137)
  	(= (slopetime-goal-passenger-at p0 n8) 39.100952)

  	(= (reward-goal-passenger-at p0 n8) 100.0)
  	(= (reward-goal-passenger-at p1 n1) 100.0)
  	(= (reward-goal-passenger-at p2 n7) 100.0)
  	(= (reward-goal-passenger-at p3 n0) 100.0)
  	(= (reward-goal-passenger-at p4 n0) 100.0)
  	(= (reward-goal-passenger-at p5 n5) 100.0)
  	(= (reward-goal-passenger-at p6 n5) 100.0)

  	(= (currentTS) 0.0)
  	(= (total-cost) 0.0)
    
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;  LTL
  
    (= (collected-goal-ticker) 0.0)
    
    ;; initial conditions
    
    (never-move-fast) ;;(always-move-slow)
    (never-move-slow) ;;(always-move-fast)
    
    (slow-allowed)    ;;(not-always-fast)
    (fast-allowed)    ;;(not-always-slow)
    
    # DEFINE YOUR PREDICATES AND FUNCTIONS HERE
    
   )

  (:goal (and
         	(collected-goal-passenger-at p0 n8)
         	(collected-goal-passenger-at p1 n1)
         	(collected-goal-passenger-at p2 n7)
         	(collected-goal-passenger-at p3 n0)
         	(collected-goal-passenger-at p4 n0)
         	(collected-goal-passenger-at p5 n5)
         	(collected-goal-passenger-at p6 n5)
         	)
         )
  (:metric minimize (total-cost))
  )
