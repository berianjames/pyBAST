subroutine transform(mu,sigma,dmu,L,theta,d0,mup,sigmap)
  implicit none
  ! Transforms bivariate (mu,sigma) to (mup,sigmap) by translation (dmu),
  !  scaling (L) and rotation (theta) about a given point (d0)

  real, dimension(2), intent(in)    :: mu
  real, dimension(2,2), intent(in)  :: sigma
  real, dimension(2), intent(in)    :: dmu
  real, dimension(2), intent(in)    :: L
  real, dimension(2), intent(in)    :: theta
  real, dimension(2), intent(in)    :: d0
  real, dimension(2), intent(out)   :: mup
  real, dimension(2,2), intent(out) :: sigmap

  real, dimension(2,2) :: U,V
  real, dimension(2,3) :: eigs
  real, dimension(2,2) :: E
  
  U = reshape( (/cos(theta),sin(theta),-sin(theta),cos(theta)/), (/2,2/) )
  mup = matmul(U,(mu*L + dmu) - d0) + d0
  
  call EV(sigma,eigs)
  E = reshape( (/L(1)*eigs(1,1),0.,0.,L(2)*eigs(2,1)/), (/2,2/) )
  V = matmul(U,eigs(:,2:3))
  sigmap = matmul(V, matmul(E,transpose(V)))

end subroutine transform

subroutine distance(mu1,sigma1,mu2,sigma2,D)
  implicit none

  real, dimension(2), intent(in)    :: mu1
  real, dimension(2,2), intent(in)  :: sigma1
  real, dimension(2), intent(in)    :: mu2
  real, dimension(2,2), intent(in)  :: sigma2
  real, intent(out)                 :: D

  real, dimension(2,2) :: S,T
  real, dimension(2) :: dmu,tmp

  S = sigma1 + sigma2
  call inv(S,T)
  dmu = mu1 - mu2

  tmp(1) = dmu(1)*T(1,1) + dmu(2)*T(1,2)
  tmp(2) = dmu(2)*T(2,1) + dmu(2)*T(2,2)
  D = (1./8.) * (tmp(1)*dmu(1) + tmp(2)*dmu(2))

end subroutine distance

subroutine EV(S,eig)
  real, dimension(2,2), intent(in) :: S
  real, dimension(2,3), intent(out) :: eig

  real :: T,D
  
  T = trace(S)
  D = det(S)

  ! Eigenvalues
  eig(1,1) = (T/2) + sqrt(T**2/4 - D) 
  eig(2,1) = (T/2) - sqrt(T**2/4 - D) 

  ! Eigenvectors

  if ( S(2,1)/=0 ) then
     eig(:,2:3) = reshape( (/eig(1,1)-S(2,2),S(2,1),eig(2,1)-S(2,2),S(2,1)/), (/2,2/) )
  elseif( S(1,2)/= 0 ) then
     eig(:,2:3) = reshape( (/S(1,2),eig(1,1)-S(1,1),S(1,2),eig(2,1)-S(1,1)/), (/2,2/) )
  else
     eig(:,2:3) = reshape((/1,0,0,1/),(/2,2/))
  end if
end subroutine EV

subroutine inv(S,T)

  real,dimension(2,2),intent(in)  :: S
  real,dimension(2,2),intent(out) :: T

  T = (1/det(S)) * reshape( (/S(2,2), -S(2,1), -S(1,2), S(1,1)/) , (/2,2/) )
  
end subroutine inv

real function det(S)
  real,dimension(2,2),intent(in)  :: S

  det = ( S(1,1)*S(2,2) - S(1,2)*S(2,1) )

end function det

real function trace(S)
  real, dimension(2,2), intent(in) :: S

  trace = S(1,1) + S(2,2)

end function trace
