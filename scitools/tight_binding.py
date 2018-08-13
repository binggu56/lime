import sys
import numpy as np
from scipy import linalg
from scipy.special import jv

class TightBinding:
    def __init__(self, Norbs, a, r, onsite, hop_inter, hop_intra, D=1):
        """
        """
        self.Norbs = Norbs # number of atomic orbitals per unit cell
        self.a = a # lattice constant
        self.r = r # coordinates of atoms shape [Norbs]
        self.onsite = onsite # [Norbs], onsite energies
        self.hop_inter = hop_inter # inter-hopping parameters, [Norbs, Norbs]
        self.hop_intra = hop_intra # intra-hopping parameters
        self.D = D # dimensionality of the problem
        if D != 1:
            sys.exit('Error: only support 1-D tight-binding model.')

    def bands(self, k):
        """
        Compute the band structure in the first Bloch wavevector k = [-pi/a, pi/a]
        """

        # onsite energies
        onsite = self.onsite
        # consider the case where hop_inter and hop_intra are different
        # the intra-cell hopping term \Sum_i a_i^\dag b_i = \Sum_k a_k^\dag b_k e^{ik(rb - ra)}
        # the inter-cell hopping term \Sum_j b_j^\dag a_{j+1} = \Sum_k b_k^\dag a_k e^{ik(ra - rb)}
        hop_intra = self.hop_intra
        hop_inter = self.hop_inter
        Norbs = self.Norbs
        r = self.r
        a = self.a

        # constuct the H_k matrix, that is the Hamiltonian expressed in terms of spacial Fourier space
        # H = (a_k, b_k)^\dag H(k) (a_k, b_k)^T
        H = np.zeros((Norbs, Norbs), dtype=np.complex128)
        # onsite energy
        for i in range(Norbs): H[i,i] = onsite[i]
        # hopping parameters
        for i in range(Norbs):
            for j in range(i+1, Norbs):

                H[i,j] = hop_intra[i,j] * np.exp(-1j * k * (r[i] - r[j])) + \
                    hop_inter[i,j] * np.exp(-1j * k *(a + r[i] - r[j]))

                H[j,i] = np.conj(H[i,j])

        eigvals = linalg.eigvals(H)

        return eigvals

    def Floquet_bands(self, k, Nt, E0, omega):
        """
        Compute the Floquet-Bloch band structure for a tight-binding model
            in the first BZ using the minimal coupling method k -> k + A(t)

        Args:

            k: wavevector in the first BZ
            E0: electric field amplitude
            omega: driving frequency

        Returns:

            eigvals:  quasiband energies at k in the first Floquet-BZ

        """
        onsite = self.onsite
        # consider the case where hop_inter and hop_intra are different
        # the intra-cell hopping term \Sum_i a_i^\dag b_i =
        #   \Sum_k a_k^\dag b_k e^{ik(rb - ra)}
        # the inter-cell hopping term
        # \Sum_j b_j^\dag a_{j+1} = \Sum_k b_k^\dag a_k e^{ik(ra - rb)}
        hop_intra = self.hop_intra
        hop_inter = self.hop_inter
        Norbs = self.Norbs
        r = self.r
        R = self.a # lattice constant
        Norbs = self.Norbs

        NF = Norbs * Nt
        F = np.zeros((NF,NF), dtype=np.complex128)

        N0 = -(Nt-1)/2 # starting point for Fourier companent of time exp(i n w t)

        # construc the Floquet H for a general tight-binding Hamiltonian
        for n in range(Nt):
            for m in range(Nt):

                # atomic basis index
                for a in range(Norbs):
                    for b in range(Norbs):

                    # map the index i to double-index (n,k) n : time Fourier component
                    # with relationship for this :  Norbs * n + k = i

                        i = Norbs * n + a
                        j = Norbs * m + b
#                # fill a block of the Floquet Matrix
#                istart = Norbs * n
#                iend = Norbs * (n+1)
#                jstart = Norbs * m
#                jend = Norbs * (m+1)
#
#                F[istart:iend, jstart:jend] = HamiltonFT(H0, H1, n-m) + (n + N0) \
#                         * omega * delta(n, m) * np.eye(Norbs)
                        z = E0/omega * (r[a] - r[b])

                        F[i,j] = onsite[a] * float(a==b) + \
                            (n+N0) * omega * float(n==m) * float(a==b)\
                            + hop_intra[a,b] * np.exp(-1j * k * (r[a] - r[b])) *\
                                     (-1j)**(m-n) * jv(m-n, z) + \
                            + hop_inter[a,b] * np.exp(-1j * k * (R + r[a] - r[b]))\
                                * (-1j)**(m-n) * jv(m-n, z)

                        #F[j,i] = np.conj(F[i,j])



        # for a two-state model
    #    for n in range(Nt):
    #        for m in range(Nt):
    #            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
    #            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
    #            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
    #            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
        #print('\n Floquet matrix \n', F)

        # compute the eigenvalues of the Floquet Hamiltonian,
        eigvals, eigvecs = linalg.eigh(F)

        # specify a range to choose the quasienergies, choose the first BZ
        # [-hbar omega/2, hbar * omega/2]
        eigvals_subset = np.zeros(Norbs)
        eigvecs_subset = np.zeros((NF , Norbs), dtype=np.complex128)


        # check if the Floquet states is complete
        j = 0
        for i in range(NF):
            if  eigvals[i] < omega/2.0 and eigvals[i] > -omega/2.0:
                eigvals_subset[j] = eigvals[i]
                eigvecs_subset[:,j] = eigvecs[:,i]
                j += 1
        if j != Norbs:
            print("Error: Number of Floquet states {} is not equal to \
                  the number of orbitals {} in the first BZ. \n".format(j, Norbs))
            sys.exit()

        return eigvals_subset, eigvecs_subset

#    def HamiltonFT(self, n, k):
#
#        H = np.zeros((Norbs, Norbs), dtype=np.complex128)
#        # onsite energy
#        for i in range(Norbs): H[i,i] = onsite[i]
#        # hopping parameters
#        for i in range(Norbs):
#            for j in range(i+1, Norbs):
#
#                H[i,j] = hop_intra[i,j] * np.exp(-1j * k * (r[i] - r[j])) + \
#                    hop_inter[i,j] * np.exp(-1j * k *(a + r[i] - r[j]))
#
#                H[j,i] = np.conj(H[i,j])
import matplotlib.pyplot as plt

def test_FloquetBloch():
    # test tight_binding module
    Norbs = 2 # number of Single-Particle (SP) orbitals in unit cell.
    # if Nb = 2, label each orbital as a and b
    a = 3.2 # in Angstroms, Lattice constant, define the first BZ = (-pi/a, pi/a)
    r = [0.0, 0.0] # coordinates of a and b


    onsite = np.array([-1.6 , 1.6])
    #hop = -1.0
    Nt = 15
    E0 = 2.0
    omega = 0.38


    hop_intra = np.zeros((Norbs, Norbs))
    hop_inter = np.zeros((Norbs, Norbs))
    hop_intra[0, 1] = 1.0
    hop_intra[1, 0] = hop_intra[0, 1]
    hop_inter[0, 1] = 1.0
    hop_inter[1, 0] = hop_inter[0, 1]

    TB = TightBinding(Norbs, a, r, onsite, hop_inter, hop_intra)

    kz = np.linspace(0, 2.*np.pi/a, 100)

    f = open('Floquet_bands.dat', 'w')
    for k in kz:
        eigvals = TB.Floquet_bands(k, Nt, E0, omega)[0]

        f.write('{} {} {} \n'.format(k, *np.real(eigvals)))

    f.close()

    fig, ax = plt.subplots(figsize=(4,4))
    k, Ev, Ec = np.genfromtxt('Floquet_bands.dat',unpack=True)
    for n in range(0,1):
        ax.plot(k, Ev + n * omega, 'k-',  label=r'$E_v(k)$')
        ax.plot(k, Ec + n * omega, 'k-',  label=r'$E_c(k)$')

# missing exact model to benchmark
#    Ec_exact = -np.sqrt(1.6**2 + (2*hop*np.cos(kz*a/2))**2)
#    Ev_exact =  np.sqrt(1.6**2 + (2*hop*np.cos(kz*a/2))**2)
#
#    ax.plot(k, Ev_exact, '--',  label=r'$E_v(k)$')
#    ax.plot(k, Ec_exact, '--',  label=r'$E_c(k)$')

    #ax.set_xlim(0,10)
    #ax.legend()
    ax.set_xlabel('k')
    ax.set_ylabel('Energy (a.u.)')
    plt.savefig('Floquet_bands.eps',dpi=1200)
    plt.draw()

#test_TightBinding()
test_FloquetBloch()